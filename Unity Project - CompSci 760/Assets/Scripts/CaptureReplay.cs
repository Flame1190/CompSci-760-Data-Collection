using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class CaptureReplay : MonoBehaviour
{
    float _time = 0;
    [SerializeField] float _maxTime = 60;

    [SerializeField] Transform _oculusLeftEye;
    [SerializeField] Transform _oculusRightEye;

    [SerializeField] Transform _replayLeftEye;
    [SerializeField] Transform _replayRightEye;

    bool _recording;
    int _frameIndex;
    bool _viewOculus = true;

    Replay _replay;

    [SerializeField] TMP_InputField _inputField;
    [SerializeField] TMP_Text _replayCountText;
    [SerializeField] TMP_Text _frameCountText;
    [SerializeField] TMP_Text _recordText;
    [SerializeField] TMP_Text _timeText;

    [SerializeField] string path;

    private void Start()
    {
        SaveAndLoad.Load();
    }

    private void Update()
    {
        _timeText.text = "Time: " + (Mathf.Round(_time * 1000) / 1000) + "/" + (Mathf.Round(_maxTime * 1000) / 1000);

        if (SaveAndLoad.Loaded())
        {
            _replayCountText.text = "Saved Replays: " + SaveAndLoad.data.GetReplayCount();
        }

        if (_replay != null)
        {
            _frameCountText.text = "Frames: " + _frameIndex + "/" + _replay.GetFrameCount();
        }

        if (_recording)
        {
            _time = Mathf.Min(_time + Time.deltaTime, _maxTime);

            if (_time >= _maxTime)
            {
                ToggleRecord();
            }
            else
            {
                _replay.AddFrameInfo(_time, _oculusLeftEye.position, _oculusLeftEye.rotation, _oculusRightEye.position, _oculusRightEye.rotation);
            }
        }
    }

    public void ToggleRecord()
    {
        _recording = !_recording;

        // Start
        if (_recording)
        {
            if (!_viewOculus) ToggleCameras();

            _recordText.text = "... Stop Recording ...";

            _time = 0;

            _replay = new Replay();
        }
        // Stop
        else
        {
            _recordText.text = "Start Recording";
        }
    }

    public void LoadReplay()
    {
        if (_recording) ToggleRecord();

        int index = int.Parse(_inputField.text);

        _replay = SaveAndLoad.data.GetReplay(index);
    }

    public void DeleteReplay()
    {
        SaveAndLoad.data.RemoveReplay(int.Parse(_inputField.text));
    }

    public void SetMaxTime()
    {
        _maxTime = int.Parse(_inputField.text);
    }

    public void LoadFrame()
    {
        LoadFrameSet(int.Parse(_inputField.text));
    }

    public void SaveReplay()
    {
        SaveAndLoad.data.AddReplay(_replay);
    }

    public void ToggleCameras()
    {
        _viewOculus = !_viewOculus;

        _oculusLeftEye.gameObject.SetActive(_viewOculus);
        _oculusRightEye.gameObject.SetActive(_viewOculus);

        _replayLeftEye.gameObject.SetActive(!_viewOculus);
        _replayRightEye.gameObject.SetActive(!_viewOculus);
    }

    public void LoadFrameAdd(int offset)
    {
        LoadFrameSet(_frameIndex + offset);
    }

    void LoadFrameSet(int newIndex)
    {
        if (_viewOculus) ToggleCameras();

        _frameIndex = newIndex;

        Replay.FrameInfo frameInfo = _replay.GetFrameInfo(_frameIndex);

        _replayLeftEye.transform.position = frameInfo.GetEyeInfo(0).Position;
        _replayLeftEye.transform.rotation = frameInfo.GetEyeInfo(0).Rotation;

        _replayRightEye.transform.position = frameInfo.GetEyeInfo(1).Position;
        _replayRightEye.transform.rotation = frameInfo.GetEyeInfo(1).Rotation;
    }
}
