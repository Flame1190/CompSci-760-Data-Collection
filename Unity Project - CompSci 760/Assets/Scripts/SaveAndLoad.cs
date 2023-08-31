using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System;

public class SaveAndLoad
{
    public static LogData data;

    static string saveToFilename = "logs.dat";

    public static void Load(string filename = "logs.dat")
    {
        if (File.Exists(Application.persistentDataPath + "/" + filename))
        {
            try
            {
                using (Stream stream = File.OpenRead(Application.persistentDataPath + "/" + filename))
                {
                    BinaryFormatter formatter = new BinaryFormatter();
                    data = (LogData)formatter.Deserialize(stream);
                }
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
        else
        {
            data = new LogData();
        }

        saveToFilename = filename;

        Debug.Log("Loaded " + filename);
    }

    public static bool Loaded()
    {
        return data != null;
    }

    public static void Save()
    {
        using (Stream stream = File.OpenWrite(Application.persistentDataPath + "/" + saveToFilename))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, data);

            Debug.Log("Saved " + saveToFilename);
        }
    }
}


[Serializable]
public class LogData
{
    List<Replay> _replays = new List<Replay>();

    public void AddReplay(Replay newSession)
    {
        _replays.Add(newSession);

        SaveAndLoad.Save();
    }

    public Replay GetReplay(int index)
    {
        return _replays[index];
    }

    public int GetReplayCount()
    {
        return _replays.Count;
    }

    public void RemoveReplay(int index)
    {
        _replays.RemoveAt(index);

        SaveAndLoad.Save();
    }
}

[Serializable]
public class Replay
{
    [Serializable]
    public class FrameInfo
    {
        float _time;
        public float Time { get { return _time; } }

        EyeInfo[] eyeInfo = new EyeInfo[2];

        [Serializable]
        public class EyeInfo
        {
            float[] _position;
            public Vector3 Position { get { return new Vector3(_position[0], _position[1], _position[2]); } }
            float[] _rotation;
            public Quaternion Rotation { get { return new Quaternion(_rotation[0], _rotation[1], _rotation[2], _rotation[3]); } }

            public EyeInfo(Vector3 position, Quaternion rotation)
            {
                _position = new float[3] { position.x, position.y, position.z };

                _rotation = new float[4] { rotation.x, rotation.y, rotation.z, rotation.w };
            }
        }

        public FrameInfo(float time, Vector3 leftPosition, Quaternion leftRotation, Vector3 rightPosition, Quaternion rightRotation)
        {
            _time = time;

            eyeInfo[0] = new EyeInfo(leftPosition, leftRotation);
            eyeInfo[1] = new EyeInfo(rightPosition, rightRotation);
        }

        public EyeInfo GetEyeInfo(int index)
        {
            return eyeInfo[index];
        }
    }

    List<FrameInfo> _frames = new List<FrameInfo>();

    public void AddFrameInfo(float time, Vector3 leftPosition, Quaternion leftRotation, Vector3 rightPosition, Quaternion rightRotation)
    {
        _frames.Add(new FrameInfo(time, leftPosition, leftRotation, rightPosition, rightRotation));
    }

    public FrameInfo GetFrameInfo(int index)
    {
        return _frames[index];
    }

    public int GetFrameCount()
    {
        return _frames.Count;
    }
}