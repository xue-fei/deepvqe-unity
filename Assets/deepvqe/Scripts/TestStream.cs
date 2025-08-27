using System;
using System.Collections.Generic;
using UnityEngine;

public class TestStream : MonoBehaviour
{
    float[] ogFloats;
    DeepVqeStream deepVqeStream;
    List<float> destList = new List<float>();

    // Start is called before the first frame update
    void Start()
    {
        deepVqeStream = new DeepVqeStream(Application.streamingAssetsPath + "/deepvqe_simple.onnx");

        ogFloats = Util.ReadWav(Application.dataPath + "/test.wav");

        Loom.RunAsync(() =>
        {
            int frameSize = 514;
            for (int i = 0; i < ogFloats.Length; i += frameSize)
            {
                int remaining = ogFloats.Length - i;
                int currentFrameSize = remaining < frameSize ? remaining : frameSize;
                float[] frame = new float[currentFrameSize];
                Array.Copy(ogFloats, i, frame, 0, currentFrameSize);
                float[] processedFrame = deepVqeStream.ProcessFrame(frame);
                destList.AddRange(processedFrame);
            }
        });
    }
    // Update is called once per frame
    void Update()
    {

    }

    private void OnDestroy()
    {
        if (deepVqeStream != null)
        {
            deepVqeStream.Dispose();
        }
        Util.SaveClip(1, 16000, destList.ToArray(), Application.dataPath + "/dest.wav");
    }
}