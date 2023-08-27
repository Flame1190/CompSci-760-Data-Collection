using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class SceneDisplaySettingsManager : MonoBehaviour
{
    [SerializeField]
    GameObject MotionVectorsCanvas;

    [SerializeField]
    Volume DepthTextureVolume;

    [SerializeField]
    bool DisplayDepthMap;

    [SerializeField]
    bool DisplayMotionVectors;

    private void Start()
    {
        MotionVectorsCanvas.SetActive(DisplayMotionVectors);
        DepthTextureVolume.gameObject.SetActive(DisplayDepthMap);
    }
}
