using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

public class DeepVqeStream : IDisposable
{
    private InferenceSession _session;
    private Dictionary<string, Tensor<float>> _cache;

    public DeepVqeStream(string modelPath)
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
        InitializeCache();
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
    }

    private DenseTensor<float> CreateZeroTensor(int[] dimensions)
    {
        var totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        var data = new float[totalSize];
        return new DenseTensor<float>(data, dimensions);
    }

    public float[] ProcessFrame(float[] frameData)
    {
        // 确保输入数据形状正确 (1, 257, 1, 2)
        if (frameData.Length != 514) // 257 * 2
        {
            throw new ArgumentException("Frame data must have 514 elements");
        }
        // 创建输入张量
        var inputTensor = new DenseTensor<float>(frameData, new[] { 1, 257, 1, 2 });

        // 准备输入
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mix", inputTensor)
        };

        // 添加所有缓存作为输入
        foreach (var cacheItem in _cache)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(cacheItem.Key, cacheItem.Value));
        }

        // 运行推理
        using (var results = _session.Run(inputs))
        {
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
            return output.ToArray();
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}