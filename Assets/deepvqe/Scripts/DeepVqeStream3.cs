using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public class DeepVqeStream3 : IDisposable
{
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private readonly InferenceSession _session;
    private Dictionary<string, Tensor<float>> _cache;
    private readonly float[] _hannWindow;

    public DeepVqeStream3(string modelPath)
    {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.AppendExecutionProvider_CPU(); // 使用CPU执行提供程序

        // 初始化ONNX会话
        _session = new InferenceSession(modelPath, options);

        // 生成汉宁窗
        _hannWindow = GenerateHannWindow(WIN_LENGTH);

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

    /// <summary>
    /// 生成汉宁窗
    /// </summary>
    private float[] GenerateHannWindow(int length)
    {
        var window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)(0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        }
        return window;
    }
      
    /// <summary>
    /// 短时傅里叶变换(STFT)
    /// </summary>
    public (Complex[,], int) Stft(float[] input)
    {
        int numFrames = 1 + (input.Length - N_FFT) / HOP_LENGTH;
        var stftResult = new Complex[N_FFT / 2 + 1, numFrames];

        for (int t = 0; t < numFrames; t++)
        {
            int start = t * HOP_LENGTH;
            int end = start + N_FFT;

            // 提取帧并应用窗函数
            var frame = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++)
            {
                if (start + i < input.Length)
                    frame[i] = input[start + i] * _hannWindow[i];
                else
                    frame[i] = 0; // 超出范围部分补零
            }

            // 转换为复数数组
            var complexFrame = new Complex[N_FFT];
            for (int i = 0; i < N_FFT; i++)
            {
                complexFrame[i] = new Complex(frame[i], 0);
            }

            // 应用FFT
            Fourier.Forward(complexFrame, FourierOptions.Matlab);

            // 只保留前半部分（实信号对称性）
            for (int f = 0; f <= N_FFT / 2; f++)
            {
                stftResult[f, t] = complexFrame[f];
            }
        }

        return (stftResult, numFrames);
    }

    /// <summary>
    /// 逆短时傅里叶变换(ISTFT)
    /// </summary>
    public float[] Istft(Complex[,] stftData, int originalLength)
    {
        int numFreqs = stftData.GetLength(0);
        int numFrames = stftData.GetLength(1);
        int outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;

        var output = new float[outputLength];
        var windowSum = new float[outputLength];

        for (int t = 0; t < numFrames; t++)
        {
            int start = t * HOP_LENGTH;

            // 重建完整的复数频谱
            var complexFrame = new Complex[N_FFT];
            for (int f = 0; f < numFreqs; f++)
            {
                complexFrame[f] = stftData[f, t];
            }

            // 利用对称性填充另一半频谱
            for (int f = numFreqs; f < N_FFT; f++)
            {
                complexFrame[f] = Complex.Conjugate(complexFrame[N_FFT - f]);
            }

            // 应用逆FFT
            Fourier.Inverse(complexFrame, FourierOptions.Matlab);

            // 转换回实数并应用窗函数
            for (int i = 0; i < N_FFT; i++)
            {
                int pos = start + i;
                if (pos < outputLength)
                {
                    output[pos] += (float)(complexFrame[i].Real * _hannWindow[i]);
                    windowSum[pos] += _hannWindow[i] * _hannWindow[i];
                }
            }
        }

        // 应用重叠相加并归一化
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-12)
            {
                output[i] /= windowSum[i];
            }
        }

        // 裁剪到原始长度
        if (originalLength < outputLength)
        {
            Array.Resize(ref output, originalLength);
        }

        return output;
    }

    /// <summary>
    /// 处理音频文件
    /// </summary>
    public float[] ProcessFrame(float[] frameData)
    { 
        // 计算STFT
        var (stftData, numFrames) = Stft(frameData);
        int numFreqs = stftData.GetLength(0);

        // 准备模型输入 (B=1, F=257, T, 2)
        float[] inputData = new float[1 * numFreqs * numFrames * 2]; 
        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numFreqs; f++)
            {
                int index = 0 * numFreqs * numFrames * 2 + f * numFrames * 2 + t * 2;
                inputData[index] = (float)stftData[f, t].Real;     // 实部
                inputData[index + 1] = (float)stftData[f, t].Imaginary;  // 虚部
            }
        }

        // 创建输入张量
        var inputTensor = new DenseTensor<float>(inputData, new[] { 1, numFreqs, numFrames, 2 });

        // 运行推理
        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("mix", inputTensor) // 注意：输入名称需要与ONNX模型匹配
            };

        // 添加所有缓存作为输入
        foreach (var cacheItem in _cache)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(cacheItem.Key, cacheItem.Value));
        }

        Console.WriteLine("开始模型推理...");
        using (var outputs = _session.Run(inputs))
        {
            // 获取输出张量
            var outputTensor = outputs.First().AsTensor<float>();

            // 转换回复数形式的STFT结果
            var enhancedStft = new Complex[numFreqs, numFrames];
            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numFreqs; f++)
                {
                    int index = 0 * numFreqs * numFrames * 2 + f * numFrames * 2 + t * 2;
                    float real = outputTensor[0, f, t, 0];
                    float imag = outputTensor[0, f, t, 1];
                    enhancedStft[f, t] = new Complex(real, imag);
                }
            }

            //执行逆STFT变换
            float[] enhancedAudio = Istft(enhancedStft, frameData.Length);
            return enhancedAudio; 
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}