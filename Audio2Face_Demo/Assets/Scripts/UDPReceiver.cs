using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System;

public class UDPReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5005;

    // 存放接收到的 60 个浮点数
    private float[] blendshapeData = new float[60];
    private SkinnedMeshRenderer skinnedMesh;
    private bool isDataNew = false;

    void Start()
    {
        skinnedMesh = GetComponent<SkinnedMeshRenderer>();
        if (skinnedMesh == null)
        {
            Debug.LogError("没有找到 SkinnedMeshRenderer！请确认脚本挂载在正确的脸部模型上。");
            return;
        }

        // --- 新增功能：打印 BlendShape 列表 ---
        //PrintBlendShapeNames();
        // ------------------------------------

        // 开启后台线程接收数据，避免卡死 Unity 主线程
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // 打印当前模型所有 BlendShape 的索引和名称
    private void PrintBlendShapeNames()
    {
        Mesh mesh = skinnedMesh.sharedMesh;
        int count = mesh.blendShapeCount;

        Debug.Log($"<color=cyan>检测到模型共有 {count} 个 BlendShapes：</color>");

        for (int i = 0; i < count; i++)
        {
            string shapeName = mesh.GetBlendShapeName(i);
            Debug.Log($"Index: {i} | Name: {shapeName}");
        }
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

        while (true)
        {
            try
            {
                // 接收二进制数据
                byte[] data = client.Receive(ref anyIP);

                // 每 4 个字节(float)解析为一个数字，总共 60 个
                for (int i = 0; i < 60; i++)
                {
                    blendshapeData[i] = BitConverter.ToSingle(data, i * 4);
                }
                isDataNew = true;
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    void Update()
    {
        if (isDataNew)
        {
            // 动态获取模型实际拥有的 BlendShape 数量，防止越界
            int actualShapeCount = skinnedMesh.sharedMesh.blendShapeCount;
            // 取我们发送的维度(60)和实际数量的最小值
            int loopCount = Mathf.Min(60, actualShapeCount);

            for (int i = 0; i < loopCount; i++)
            {
                // 注意：这里 i 可能需要 +1 避开 Basis 键，你先直接跑跑看
                skinnedMesh.SetBlendShapeWeight(i, blendshapeData[i]);
            }
            isDataNew = false;
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (client != null) client.Close();
    }
}