using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class ScreenshotTaker : MonoBehaviour
{
    public Camera targetCamera;
    public string outputName;

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.I))
        {
            //print(Application.dataPath + "/cameracapture.png");
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SaveCameraView(targetCamera);
        }
    }

    void SaveCameraView(Camera cam)
    {
        int width = XRSettings.eyeTextureWidth;
        int height = XRSettings.eyeTextureHeight;
        print(width + ", " + height);

        RenderTexture screenTexture = new RenderTexture(width, height, 16);
        cam.targetTexture = screenTexture;
        RenderTexture.active = screenTexture;
        cam.Render();
        Texture2D renderedTexture = new Texture2D(width, height);
        renderedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        RenderTexture.active = null;
        byte[] byteArray = renderedTexture.EncodeToPNG();
        System.IO.File.WriteAllBytes(Application.dataPath + "/Screenshots/" + outputName + ".png", byteArray);
        cam.targetTexture = null;
    }
}


