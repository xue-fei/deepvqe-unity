using UnityEngine;

public class TestStream : MonoBehaviour
{
    DeepVqeStream4 deepVqeStream;

    // Start is called before the first frame update
    void Start()
    {
        deepVqeStream = new DeepVqeStream4(Application.streamingAssetsPath + "/deepvqe.onnx");

        float[] longAudio = Util.ReadWav(Application.dataPath + "/test.wav");

        float[] enhancedAudio = new float[longAudio.Length];
        int count = deepVqeStream.ProcessFrame(longAudio, out enhancedAudio);
        if (count > 0)
        {
            Util.SaveClip(1, 16000, enhancedAudio, Application.dataPath + "/dest.wav");
        } 
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
    }
}