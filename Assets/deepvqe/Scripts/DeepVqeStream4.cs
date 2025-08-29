using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public class DeepVqeStream4 : IDisposable
{
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    /// <summary>
    /// 257
    /// </summary>
    private const int NUM_BINS = N_FFT / 2 + 1;

    private InferenceSession _session;
    private Dictionary<string, Tensor<float>> _cache;
    private readonly List<NamedOnnxValue> _inputTensors = new List<NamedOnnxValue>();
    private readonly DenseTensor<float> _inputMixTensor;
    private struct AudioBuffer
    {
        public float[] All; // 总长度N_FFT=512
        public ArraySegment<float> Overlap; // 前HOP_LENGTH=256（重叠保留区）
        public ArraySegment<float> Input;   // 后HOP_LENGTH=256（新数据缓存区）
    }
    private readonly AudioBuffer _audioBuffer;
    private int _inputBufferSize; // Input区有效数据量

    // ISTFT重叠相加状态 
    private readonly float[] _outFramesHop;
    // STFT/ISTFT组件
    private readonly float[] _window;
    private readonly float[] _windowDivNfft;
    private readonly Complex[] _fftBuffer;
    private readonly Complex[] _ifftBuffer;

    public DeepVqeStream4(string modelPath)
    {
        // 设置ONNX运行时选项
        var options = new SessionOptions
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = 1,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            //EnableProfiling = true,
        };

        // 根据需求选择CPU或GPU执行提供程序
        options.AppendExecutionProvider_CPU();
        // 或者使用GPU（如果可用）
        // options.AppendExecutionProvider_CUDA(0);

        _session = new InferenceSession(modelPath, options);


        // 初始化输入张量列表（动态更新缓存引用）
        _inputMixTensor = new DenseTensor<float>(dimensions: new[] { 1, NUM_BINS, 1, 2 });
        InitializeCache();

        // 音频缓冲区 
        _audioBuffer.All = new float[N_FFT];
        _audioBuffer.Overlap = new ArraySegment<float>(_audioBuffer.All, 0, HOP_LENGTH);
        _audioBuffer.Input = new ArraySegment<float>(_audioBuffer.All, HOP_LENGTH, HOP_LENGTH);
        _inputBufferSize = 0;

        // 重叠相加状态缓存
        _outFramesHop = new float[HOP_LENGTH];

        // 窗函数 
        _window = CreateHannWindow(WIN_LENGTH);
        _windowDivNfft = new float[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
        {
            _windowDivNfft[i] = _window[i] / N_FFT;
        }
        // FFT/IFFT缓冲区
        _fftBuffer = new Complex[N_FFT];
        _ifftBuffer = new Complex[N_FFT];
    }

    private void InitializeCache()
    {
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

        _inputTensors.Add(NamedOnnxValue.CreateFromTensor("mix", _inputMixTensor));
        // 添加所有缓存作为输入
        foreach (var cacheItem in _cache)
        {
            _inputTensors.Add(NamedOnnxValue.CreateFromTensor(cacheItem.Key, cacheItem.Value));
        }
    }

    private DenseTensor<float> CreateZeroTensor(int[] dimensions)
    {
        var totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        var data = new float[totalSize];
        return new DenseTensor<float>(data, dimensions);
    }

    float[] stftInput = new float[N_FFT];
    Complex[] enhSpectrum = new Complex[NUM_BINS];
    Complex[] stftResult;
    float[] istftOutput;

    public int ProcessFrame(float[] frameData, out float[] outputSamples)
    {
        outputSamples = Array.Empty<float>();
        int numSamples = frameData.Length;
        // 输入不足一帧（256样本）时缓存
        if (numSamples + _inputBufferSize < HOP_LENGTH)
        {
            Array.Copy(frameData, 0,
                      _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                      numSamples);
            _inputBufferSize += numSamples;
            return 0;
        }
        // 计算可处理的帧数
        int totalAvailable = numSamples + _inputBufferSize;
        int numFrames = totalAvailable / HOP_LENGTH;
        int outputSize = numFrames * HOP_LENGTH;
        outputSamples = new float[outputSize];

        // 复制上一次的重叠数据到输出 
        Array.Copy(_outFramesHop, outputSamples, HOP_LENGTH);

        int inputOffset = 0; // 输入数据处理偏移量
        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            // 1. 填充输入缓冲区（补充至256样本）
            int need = HOP_LENGTH - _inputBufferSize;
            if (need > 0)
            {
                int copy = Math.Min(need, numSamples - inputOffset);
                Array.Copy(frameData, inputOffset,
                          _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                          copy);
                _inputBufferSize += copy;
                inputOffset += copy;
            }

            // 2. 准备STFT输入（overlap + input，共512样本）

            Array.Copy(_audioBuffer.Overlap.Array, _audioBuffer.Overlap.Offset, stftInput, 0, HOP_LENGTH);
            Array.Copy(_audioBuffer.Input.Array, _audioBuffer.Input.Offset, stftInput, HOP_LENGTH, HOP_LENGTH);

            // 应用窗函数
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                stftInput[i] *= _window[i];
            }
            // 3. 执行STFT 
            stftResult = STFT(stftInput);

            // 4. 填充ONNX输入张量（实部+虚部）
            for (int bin = 0; bin < NUM_BINS; bin++)
            {
                _inputMixTensor[0, bin, 0, 0] = (float)stftResult[bin].Real;
                _inputMixTensor[0, bin, 0, 1] = (float)stftResult[bin].Imaginary;
            }

            // 运行推理
            using var results = _session.Run(_inputTensors);

            // 获取输出
            var output = results.FirstOrDefault(r => r.Name == "enh")?.AsTensor<float>();
            if (output == null)
            {
                throw new InvalidOperationException("No output named 'enh' found");
            }
            // 更新缓存
            foreach (var result in results)
            {
                if (result.Name.EndsWith("_out") && _cache.ContainsKey(result.Name.Replace("_out", "")))
                {
                    _cache[result.Name.Replace("_out", "")] = result.AsTensor<float>().ToDenseTensor();
                }
            }
            // 7. 转换增强结果为复数频谱 
            for (int bin = 0; bin < NUM_BINS; bin++)
            {
                enhSpectrum[bin] = new Complex(output[0, bin, 0, 0], output[0, bin, 0, 1]);
            }
            // 8. 执行ISTFT 
            istftOutput = ISTFT(enhSpectrum);

            // 9. 重叠相加 
            // 前256样本叠加到输出
            for (int i = 0; i < HOP_LENGTH; i++)
            {
                outputSamples[frameIdx * HOP_LENGTH + i] += istftOutput[i] * _windowDivNfft[i];
            }
            // 非最后一帧：叠加256-512样本到输出
            if (frameIdx < numFrames - 1)
            {
                for (int i = HOP_LENGTH; i < WIN_LENGTH; i++)
                {
                    int outputPos = frameIdx * HOP_LENGTH + i;
                    if (outputPos < outputSize)
                    {
                        outputSamples[outputPos] += istftOutput[i] * _windowDivNfft[i];
                    }
                }
            }
            // 最后一帧：保存256-512样本作为下次overlap
            else
            {
                for (int i = 0; i < HOP_LENGTH; i++)
                {
                    _outFramesHop[i] = istftOutput[HOP_LENGTH + i] * _windowDivNfft[HOP_LENGTH + i];
                }
            } 
            //10.更新overlap（将当前input移至overlap）
            Array.Copy(_audioBuffer.Input.Array, _audioBuffer.Input.Offset,
                      _audioBuffer.Overlap.Array, _audioBuffer.Overlap.Offset,
                      HOP_LENGTH);
            _inputBufferSize = 0;
        }

        // 缓存剩余输入数据
        int remaining = numSamples - inputOffset;
        if (remaining > 0)
        {
            Array.Copy(frameData, inputOffset,
                      _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                      remaining);
            _inputBufferSize += remaining;
        }
        return outputSize;
    }

    /// <summary>
    /// 创建Hann窗 
    /// </summary>
    private float[] CreateHannWindow(int length)
    {
        float[] window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)Math.Sqrt(0.5f * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        }
        return window;
    }

    Complex[] result = new Complex[NUM_BINS];
    /// <summary>
    /// 执行STFT 
    /// </summary>
    private Complex[] STFT(float[] input)
    {
        // 填充FFT缓冲区（实部输入）
        for (int i = 0; i < N_FFT; i++)
        {
            _fftBuffer[i] = new Complex(input[i], 0);
        }
        // 执行FFT
        FFT(_fftBuffer, N_FFT);

        // 提取正频率分量（257个频段）

        Array.Copy(_fftBuffer, result, NUM_BINS);
        return result;
    }

    /// <summary>
    /// FFT计算 
    /// </summary>
    private void FFT(Complex[] buffer, int length)
    {
        Fourier.Forward(buffer, FourierOptions.Matlab); // 与C的kiss_fft对齐
    }

    float[] result2 = new float[N_FFT];
    /// <summary>
    /// 执行ISTFT 
    /// </summary>
    private float[] ISTFT(Complex[] spectrum)
    {
        // 填充完整频谱（含负频率共轭）
        Array.Copy(spectrum, _ifftBuffer, NUM_BINS);
        for (int i = 1; i < NUM_BINS - 1; i++)
        {
            _ifftBuffer[N_FFT - i] = Complex.Conjugate(_ifftBuffer[i]);
        }
        // 奈奎斯特分量强制为实数 
        if (N_FFT % 2 == 0)
        {
            _ifftBuffer[N_FFT / 2] = new Complex(_ifftBuffer[N_FFT / 2].Real, 0);
        }
        // 执行IFFT
        IFFT(_ifftBuffer, N_FFT);

        // 提取实部 
        for (int i = 0; i < N_FFT; i++)
        {
            result2[i] = (float)_ifftBuffer[i].Real;
        }
        return result2;
    }

    /// <summary>
    /// IFFT计算（模仿C的kiss_fft逆变换）
    /// </summary>
    private void IFFT(Complex[] buffer, int length)
    {
        // 共轭后FFT等价于IFFT 
        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }
        FFT(buffer, length);

        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}