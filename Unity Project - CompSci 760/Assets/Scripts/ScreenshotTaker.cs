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
            Test(targetCamera);
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SaveCameraView(targetCamera);
        }
    }

    void SaveCameraView(Camera cam)
    {
        // 1824, 1840

        int width = XRSettings.eyeTextureWidth;
        int height = XRSettings.eyeTextureHeight;

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

    void Test(Camera cam)
    {
        int width = XRSettings.eyeTextureWidth;
        int height = XRSettings.eyeTextureHeight;

        RenderTexture screenTexture = new RenderTexture(width, height, 16);
        cam.targetTexture = screenTexture;
        RenderTexture.active = screenTexture;
        cam.Render();
        Texture2D renderedTexture = new Texture2D(width, height);
        renderedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        RenderTexture.active = null;


        Color pixelColour = renderedTexture.GetPixel(width / 2, height / 2);
        Vector3Int pixelInts = new Vector3Int((int)(pixelColour.r * 255), (int)(pixelColour.g * 255), (int)(pixelColour.b * 255));
        Debug.LogError(pixelInts);


        cam.targetTexture = null;
    }
}