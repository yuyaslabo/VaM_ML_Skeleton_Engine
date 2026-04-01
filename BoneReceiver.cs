using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;

public class BoneRemoteReceiver : MVRScript
{
    private UdpClient _udpClient;
    private JSONStorableFloat _portParam;
    private bool _hasReceivedData = false;
    
    private Dictionary<string, Transform> _boneMap = new Dictionary<string, Transform>();
    private Dictionary<string, Vector3> _targetPos = new Dictionary<string, Vector3>();
    private Dictionary<string, Quaternion> _targetRot = new Dictionary<string, Quaternion>();

    private readonly string[] _targetIDs = {
        "hipControl","chestControl", "headControl",
        "rHandControl","lHandControl","rFootControl","lFootControl",
        "rKneeControl", "lKneeControl"
    };

    public override void Init()
    {
        // 1. JSONStorableFloatのコンストラクタで、最後の引数「constrained」を true にする
        // これにより、内部的に整数として扱われやすくなります
        _portParam = new JSONStorableFloat("Port Number", 9998f, 1024f, 65535f, true, true);
        _portParam.setCallbackFunction = (float val) => RestartUDP();
        
        // 2. スライダーを作成した後、ステップ値を 1.0 に設定する
        UIDynamicSlider slider = CreateSlider(_portParam);
        if (slider != null)
        {
            slider.valueFormat = "F0"; // 小数点以下を表示しないフォーマット
            slider.quickButtonsEnabled = true; // 必要に応じて
        }
        
        RegisterFloat(_portParam);

        HashSet<string> targetSet = new HashSet<string>(_targetIDs);

        if (containingAtom != null && containingAtom.freeControllers != null)
        {
            foreach (var ctrl in containingAtom.freeControllers)
            {
                if (ctrl == null) continue;
                string id = ctrl.name;

                if (targetSet.Contains(id))
                {
                    _boneMap[id] = ctrl.transform;
                    _targetPos[id] = ctrl.transform.localPosition;
                    _targetRot[id] = ctrl.transform.localRotation;
                    
                    if (id == "rHandControl" || id == "lHandControl")
                    {
                        JSONStorableFloat rotForce = ctrl.GetFloatJSONParam("holdRotationMaxForce");
                        if (rotForce != null) rotForce.val = 50f;
                        JSONStorableFloat rotSpring = ctrl.GetFloatJSONParam("holdRotationSpring");
                        if (rotSpring != null) rotSpring.val = 50f;
                    }
                    ctrl.currentPositionState = FreeControllerV3.PositionState.On;
                    ctrl.currentRotationState = FreeControllerV3.RotationState.On;
                }
                else
                {
                    ctrl.currentPositionState = FreeControllerV3.PositionState.Off;
                    ctrl.currentRotationState = FreeControllerV3.RotationState.Off;
                }
            }
        }

        RestartUDP();
    }

    private void RestartUDP()
    {
        CloseUDP();
        try {
            int port = (int)_portParam.val;
            _udpClient = new UdpClient(port);
            _udpClient.BeginReceive(new AsyncCallback(ReceiveCallback), null);
            SuperController.LogMessage("UDP Receiver: Listening on port " + port);
        } catch (Exception e) {
            SuperController.LogError("UDP Error: " + e.Message);
        }
    }

    private void ReceiveCallback(IAsyncResult ar)
    {
        try {
            if (_udpClient == null) return;

            IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, (int)_portParam.val);
            byte[] data = _udpClient.EndReceive(ar, ref remoteEP);
            string msg = Encoding.UTF8.GetString(data);

            string[] bones = msg.Split('|');
            foreach (var b in bones) {
                string[] d = b.Split(',');
                if (d.Length == 8) {
                    string id = d[0];
                    if (_boneMap.ContainsKey(id)) {
                        _targetPos[id] = new Vector3(float.Parse(d[1]), float.Parse(d[2]), float.Parse(d[3]));
                        _targetRot[id] = new Quaternion(float.Parse(d[4]), float.Parse(d[5]), float.Parse(d[6]), float.Parse(d[7]));
                    }
                }
            }
            _hasReceivedData = true;
            _udpClient.BeginReceive(new AsyncCallback(ReceiveCallback), null);
        } catch (Exception) {
            // エラー時は静かに閉じる
        }
    }
    
    void Update()
    {
        if (!_hasReceivedData || containingAtom == null || containingAtom.mainController == null) return;

        Transform rootT = containingAtom.mainController.transform;

        foreach (var id in _targetIDs)
        {
            if (!_boneMap.ContainsKey(id)) continue;

            Vector3 worldTargetPos = rootT.TransformPoint(_targetPos[id]);
            _boneMap[id].position = Vector3.Lerp(_boneMap[id].position, worldTargetPos, Time.deltaTime * 15f);

            Quaternion worldTargetRot = rootT.rotation * _targetRot[id];
            _boneMap[id].rotation = Quaternion.Slerp(_boneMap[id].rotation, worldTargetRot, Time.deltaTime * 15f);
        }
    }

    private void CloseUDP()
    {
        if (_udpClient != null) {
            _udpClient.Close();
            _udpClient = null;
        }
    }

    void OnDestroy() { 
        CloseUDP();
    }
}
