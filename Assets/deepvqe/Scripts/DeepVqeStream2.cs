using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public class DeepVqeStream2 : IDisposable
{
    private readonly InferenceSession _session;
    private Dictionary<string, Tensor<float>> _cache;
    private readonly int N_FFT = 512;
    private readonly int HOP_LENGTH = 256;
    private readonly int WIN_LENGTH = 512;
    private readonly float[] _window;

    public DeepVqeStream2(string modelPath)
    {
        // 设置ONNX运行时选项
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.AppendExecutionProvider_CPU(); // 使用CPU执行提供程序

        _session = new InferenceSession(modelPath, options);
        _window = CreateHannWindow(WIN_LENGTH);

        _cache = new Dictionary<string, Tensor<float>>
        {
            {"en_conv_cache1", CreateZeroTensor(new[] {1, 2, 3, 257})},
            {"en_res_cache1", CreateZeroTensor(new[] {1, 64, 3, 129})},
            {"en_conv_cache2", CreateZeroTensor(new[] {1, 64, 3, 129})},
            {"en_res_cache2", CreateZeroTensor(new[] {1, 128, 3, 65})},
            {"en_conv_cache3", CreateZeroTensor(new[] {1, 128, 3, 65})},
            {"en_res_cache3", CreateZeroTensor(new[] {1, 128, 3, 33})},
            {"en_conv_cache4", CreateZeroTensor(new[] {1, 128, 3, 33})},
            {"en_res_cache4", CreateZeroTensor(new[] {1, 128, 3, 17})},
            {"en_conv_cache5", CreateZeroTensor(new[] {1, 128, 3, 17})},
            {"en_res_cache5", CreateZeroTensor(new[] {1, 128, 3, 9})},
            {"h_cache", CreateZeroTensor(new[] {1, 1, 64 * 9})},
            {"de_conv_cache5", CreateZeroTensor(new[] {1, 128, 3, 9})},
            {"de_res_cache5", CreateZeroTensor(new[] {1, 128, 3, 9})},
            {"de_conv_cache4", CreateZeroTensor(new[] {1, 128, 3, 17})},
            {"de_res_cache4", CreateZeroTensor(new[] {1, 128, 3, 17})},
            {"de_conv_cache3", CreateZeroTensor(new[] {1, 128, 3, 33})},
            {"de_res_cache3", CreateZeroTensor(new[] {1, 128, 3, 33})},
            {"de_conv_cache2", CreateZeroTensor(new[] {1, 128, 3, 65})},
            {"de_res_cache2", CreateZeroTensor(new[] {1, 128, 3, 65})},
            {"de_conv_cache1", CreateZeroTensor(new[] {1, 64, 3, 129})},
            {"de_res_cache1", CreateZeroTensor(new[] {1, 64, 3, 129})},
            {"m_cache", CreateZeroTensor(new[] {1, 257, 2, 2})}
        };
    }

    private DenseTensor<float> CreateZeroTensor(int[] dimensions)
    {
        var totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        var data = new float[totalSize];
        return new DenseTensor<float>(data, dimensions);
    }

    public float[] ProcessFrame(float[] frameData)
    {
        int origLen = frameData.Length;

        // 计算STFT
        var stftResult = ComputeSTFT(frameData);

        // 准备模型输入
        var inputTensor = new DenseTensor<float>(frameData, new[] { 1, 257, 1, 2 });
        // 运行模型推理
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mix", inputTensor)
        };

        // 添加所有缓存作为输入
        foreach (var cacheItem in _cache)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(cacheItem.Key, cacheItem.Value));
        }

        Complex[,] enhancedStft;

        using (var results = _session.Run(inputs))
        {
            var output = results.FirstOrDefault(r => r.Name == "enh")?.AsTensor<float>();
            if (output == null)
                throw new InvalidOperationException("No output named 'enh' found");

            // 更新缓存
            foreach (var result in results)
            {
                if (result.Name.EndsWith("_out") && _cache.ContainsKey(result.Name.Replace("_out", "")))
                {
                    _cache[result.Name.Replace("_out", "")] = result.AsTensor<float>().ToDenseTensor();
                }
            }

            // 提取输出并转换为复数
            enhancedStft = new Complex[numFrames, numBins];

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    enhancedStft[t, f] = new Complex(
                        output[0, f, t, 0],
                        output[0, f, t, 1]
                    );
                }
            }
        }

        // 计算ISTFT
        var enhancedAudio = ComputeISTFT(enhancedStft);

        // 确保长度与原始音频相同
        if (enhancedAudio.Length > origLen)
        {
            enhancedAudio = enhancedAudio.Take(origLen).ToArray();
        }
        else if (enhancedAudio.Length < origLen)
        {
            var padded = new float[origLen];
            Array.Copy(enhancedAudio, padded, enhancedAudio.Length);
            enhancedAudio = padded;
        }

        return enhancedAudio;
    }

    int numFrames;
    int numBins;

    private Complex[,] ComputeSTFT(float[] audioData)
    {
        numFrames = (int)Math.Ceiling((audioData.Length - N_FFT) / (float)HOP_LENGTH) + 1;
        numBins = N_FFT / 2; // 256 for 512-point FFT

        UnityEngine.Debug.LogWarning("numFrames:" + numFrames);
        UnityEngine.Debug.LogWarning("numBins:" + numBins);

        var stftResult = new Complex[numFrames, numBins];

        for (int i = 0; i < numFrames; i++)
        {
            int start = i * HOP_LENGTH;
            var frame = new Complex[N_FFT];

            // 提取帧并应用窗口
            for (int j = 0; j < N_FFT; j++)
            {
                int idx = start + j;
                float value = (idx < audioData.Length) ? audioData[idx] * _window[j] : 0;
                frame[j] = new Complex(value, 0);
            }

            // 计算FFT
            Fourier.Forward(frame);

            // 只保留前numBins个频率bin (对称性)
            for (int f = 0; f < numBins; f++)
            {
                stftResult[i, f] = frame[f];
            }
        }

        return stftResult;
    }

    private float[] ComputeISTFT(Complex[,] stftData)
    {
        int numFrames = stftData.GetLength(0);
        int numBins = stftData.GetLength(1);
        int outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        var output = new float[outputLength];
        var overlap = new float[outputLength];

        for (int i = 0; i < numFrames; i++)
        {
            int start = i * HOP_LENGTH;
            var frame = new Complex[N_FFT];

            // 重建完整频谱 (利用对称性)
            for (int f = 0; f < numBins; f++)
            {
                frame[f] = stftData[i, f];
            }

            for (int f = numBins; f < N_FFT; f++)
            {
                frame[f] = Complex.Conjugate(frame[N_FFT - f]);
            }

            // 计算IFFT
            Fourier.Inverse(frame);

            // 应用窗口并添加到输出
            for (int j = 0; j < N_FFT; j++)
            {
                int idx = start + j;
                if (idx < outputLength)
                {
                    // 重叠相加
                    output[idx] += (float)(frame[j].Real * _window[j]);
                    overlap[idx] += _window[j] * _window[j];
                }
            }
        }

        // 归一化重叠部分
        for (int i = 0; i < outputLength; i++)
        {
            if (overlap[i] > 1e-12)
            {
                output[i] /= overlap[i];
            }
        }

        return output;
    }

    private float[] CreateHannWindow(int size)
    {
        var window = new float[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = (float)(0.5f * (1 - Math.Cos(2 * Math.PI * i / (size - 1))));
        }
        return window;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}