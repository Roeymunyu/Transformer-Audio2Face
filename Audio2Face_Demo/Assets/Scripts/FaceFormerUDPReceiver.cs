using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class FaceFormerUDPReceiver : MonoBehaviour
{
    [Header("绑定的面部管理器")]
    public FaceBlendshapesManager faceManager;

    [Header("网络配置")]
    public int port = 5005;

    // 网络与多线程组件
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning = false;

    // 线程安全的数据交换区
    private float[] latestBlendshapes = new float[52];
    private bool hasNewData = false;
    private readonly object dataLock = new object();

    private readonly string[] arkitNames = new string[52] {
        "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
        "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
        "jawForward", "jawLeft", "jawRight", "jawOpen",
        "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
        "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight", "tongueOut"
    };

    void Start()
    {
        if (faceManager == null) faceManager = GetComponent<FaceBlendshapesManager>();

        StartUDPListener();
    }

    private void StartUDPListener()
    {
        try
        {
            udpClient = new UdpClient(port);
            isRunning = true;

            // 开启后台子线程专门用来死循环监听 UDP 数据，不卡死 Unity 主渲染线程
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log($" [UDP] 成功启动！正在监听端口 {port}...");
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDP] 启动失败，端口可能被占用: {e.Message}");
        }
    }

    /// <summary>
    /// 后台线程：只负责收数据、解包
    /// </summary>
    private void ReceiveData()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, port);

        while (isRunning)
        {
            try
            {
                // Receive 会阻塞线程，直到接收到数据
                byte[] data = udpClient.Receive(ref anyIP);

                // 检查数据长度：52个 float * 每个 float 占 4 字节 = 208 字节
                if (data.Length == 208)
                {
                    // 加锁，防止主线程正读到一半被新数据覆盖
                    lock (dataLock)
                    {
                        for (int i = 0; i < 52; i++)
                        {
                            // 使用小端序解析 4 个字节为一个 float
                            latestBlendshapes[i] = BitConverter.ToSingle(data, i * 4);
                        }
                        hasNewData = true;
                    }
                }
                else
                {
                    Debug.LogWarning($" [UDP] 收到异常长度数据包: {data.Length} 字节，预期应为 208 字节 (52个float)。");
                }
            }
            catch (SocketException)
            {
                if (isRunning) Debug.LogWarning(" [UDP] Socket 被意外强制关闭。");
            }
        }
    }

    /// <summary>
    /// Unity 主线程：每帧调用，负责安全地渲染模型
    /// </summary>
    void Update()
    {
        if (hasNewData && faceManager != null)
        {
            lock (dataLock)
            {
                // 遍历 52 个表情
                for (int i = 0; i < 52; i++)
                {
                    // 应该是 0~100 的数值
                    faceManager.SetBlendshape(arkitNames[i], latestBlendshapes[i]);
                }

                hasNewData = false;
            }
        }
    }

    /// <summary>
    /// 关闭游戏时必须干掉后台线程，否则会在后台变成僵尸进程占用端口
    /// </summary>
    void OnApplicationQuit()
    {
        isRunning = false;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }

        if (udpClient != null)
        {
            udpClient.Close();
        }

        Debug.Log(" [UDP] 端口已释放，线程已安全退出。");
    }
}