using UnityEngine;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
#endif

// 定义一个可以在 Inspector 面板可视化的结构体，用来配置限制
[System.Serializable]
public struct BlendshapeLimit
{
    public string shapeName; // 例如 "jawOpen"
    [Range(0f, 100f)]
    public float maxAllowedWeight; // 例如 30f
}

// 联动配置结构体
[System.Serializable]
public struct BlendshapeLinkage
{
    [Tooltip("当接收到这个表情时 (如 mouthFunnel)")]
    public string sourceShape;
    [Tooltip("联动触发的目标表情 (如 う 或 vrc.v_ou)")]
    public string targetShape;
    [Tooltip("权重乘数 (例如 3，即触发 3 倍的 targetShape 权重)")]
    public float multiplier;
}

// [新增] 修正混合配置结构体
[System.Serializable]
public struct BlendshapeCorrective
{
    [Tooltip("驱动修正的表情 (如 jawOpen)，当它过高时触发修正")]
    public string driverShape;

    [Tooltip("被修正（压制）的表情 (如 mouthFunnel)")]
    public string affectedShape;

    [Tooltip("驱动表情超过此阈值后才开始修正 (0-100)。例如 20 表示 jawOpen > 20 时才开始压 mouthFunnel")]
    [Range(0f, 100f)]
    public float driverThreshold;

    [Tooltip("修正强度 (0-1)。1 = 驱动到 100 时完全压制，0.7 = 驱动到 100 时压到原值的 30%")]
    [Range(0f, 1f)]
    public float correctionStrength;
}

public class FaceBlendshapesManager : MonoBehaviour
{
    [Header("绑定的目标网格 (拖入脸、牙齿、睫毛等)")]
    public SkinnedMeshRenderer[] targetMeshes;

    [Header("安全限制锁 (防止破相)")]
    [Tooltip("在这里添加容易穿模的表情，比如 jawOpen 限制到 30")]
    public List<BlendshapeLimit> safeLimits;

    [Header("联动表情映射 (优化特定口型表现)")]
    [Tooltip("例如：发 mouthFunnel 时，联动触发 う，乘数写 3")]
    public List<BlendshapeLinkage> shapeLinkages;

    [Header("修正混合 Corrective Blendshape (防止组合变形过度)")]
    [Tooltip("例如：jawOpen 过高时，自动压制 mouthFunnel，避免嘴型变成夸张椭圆")]
    public List<BlendshapeCorrective> correctives;

    // 内部用来快速查找限制的字典
    private Dictionary<string, float> limitDictionary;

    // 内部用来快速查找联动规则的字典
    private Dictionary<string, List<BlendshapeLinkage>> linkageDictionary;

    // [新增] 修正规则字典: 以被影响表情为 key，查找所有影响它的修正规则
    private Dictionary<string, List<BlendshapeCorrective>> correctiveByAffected;

    // [新增] 修正规则字典: 以驱动表情为 key，查找它驱动了哪些修正规则
    private Dictionary<string, List<BlendshapeCorrective>> correctiveByDriver;

    // [新增] 存储每个表情最后一次通过 SetBlendshape 传入的原始权重
    // 这样当驱动表情变化时，可以用被影响表情的原始值重新计算修正结果
    private Dictionary<string, float> rawValues = new Dictionary<string, float>();

    void Start()
    {
        InitDictionary();
    }

    /// <summary>
    /// 初始化字典缓存，提升运行时的查询效率
    /// </summary>
    private void InitDictionary()
    {
        // 1. 初始化安全限制字典
        limitDictionary = new Dictionary<string, float>();
        if (safeLimits != null)
        {
            foreach (var limit in safeLimits)
            {
                if (!limitDictionary.ContainsKey(limit.shapeName))
                {
                    limitDictionary.Add(limit.shapeName, limit.maxAllowedWeight);
                }
            }
        }

        // 2. 初始化联动映射字典
        linkageDictionary = new Dictionary<string, List<BlendshapeLinkage>>();
        if (shapeLinkages != null)
        {
            foreach (var link in shapeLinkages)
            {
                if (!linkageDictionary.ContainsKey(link.sourceShape))
                {
                    linkageDictionary[link.sourceShape] = new List<BlendshapeLinkage>();
                }
                linkageDictionary[link.sourceShape].Add(link);
            }
        }

        // 3. 初始化修正混合字典
        correctiveByAffected = new Dictionary<string, List<BlendshapeCorrective>>();
        correctiveByDriver = new Dictionary<string, List<BlendshapeCorrective>>();
        if (correctives != null)
        {
            foreach (var c in correctives)
            {
                // 以 affectedShape 为 key
                if (!correctiveByAffected.ContainsKey(c.affectedShape))
                    correctiveByAffected[c.affectedShape] = new List<BlendshapeCorrective>();
                correctiveByAffected[c.affectedShape].Add(c);

                // 以 driverShape 为 key
                if (!correctiveByDriver.ContainsKey(c.driverShape))
                    correctiveByDriver[c.driverShape] = new List<BlendshapeCorrective>();
                correctiveByDriver[c.driverShape].Add(c);
            }
        }
    }

    /// <summary>
    /// 获取某个表情的权重上限
    /// </summary>
    private float GetMaxWeightLimit(string shapeName)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            if (safeLimits != null)
            {
                foreach (var limit in safeLimits)
                {
                    if (limit.shapeName == shapeName) return limit.maxAllowedWeight;
                }
            }
            return 100f;
        }
#endif
        if (limitDictionary == null) InitDictionary();
        if (limitDictionary.TryGetValue(shapeName, out float maxWeight))
        {
            return maxWeight;
        }
        return 100f;
    }

    /// <summary>
    /// 获取某个表情缓存的原始权重值
    /// </summary>
    private float GetRawValue(string shapeName)
    {
        if (rawValues != null && rawValues.TryGetValue(shapeName, out float val))
            return val;
        return 0f;
    }

    /// <summary>
    /// [新增] 计算某个被影响表情的修正系数 (0~1)
    /// 返回 1 表示无修正，返回 0 表示完全压制
    /// 可以有多个驱动源同时影响同一个表情，修正系数会叠乘
    /// </summary>
    private float ComputeCorrectiveFactor(string affectedShape)
    {
        float factor = 1f;

#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            // 编辑器模式下直接遍历 List，方便实时调试
            if (correctives != null)
            {
                foreach (var c in correctives)
                {
                    if (c.affectedShape == affectedShape)
                    {
                        float range = 100f - c.driverThreshold;
                        if (range <= 0f) continue; // 阈值为 100 时不修正

                        float driverRaw = GetRawValue(c.driverShape);
                        if (driverRaw > c.driverThreshold)
                        {
                            float t = Mathf.Clamp01((driverRaw - c.driverThreshold) / range);
                            factor *= (1f - t * c.correctionStrength);
                        }
                    }
                }
            }
            return Mathf.Clamp01(factor);
        }
#endif

        // 运行时使用字典快速查询
        if (correctiveByAffected == null) InitDictionary();
        if (correctiveByAffected.TryGetValue(affectedShape, out var rules))
        {
            foreach (var c in rules)
            {
                float range = 100f - c.driverThreshold;
                if (range <= 0f) continue;

                float driverRaw = GetRawValue(c.driverShape);
                if (driverRaw > c.driverThreshold)
                {
                    float t = Mathf.Clamp01((driverRaw - c.driverThreshold) / range);
                    factor *= (1f - t * c.correctionStrength);
                }
            }
        }

        return Mathf.Clamp01(factor);
    }

    /// <summary>
    /// [新增] 当某个驱动表情变化时，重新计算并应用它所影响的所有被修正表情
    /// 例如: jawOpen 变化后，重新修正 mouthFunnel 的权重
    /// </summary>
    private void ReapplyAffectedByDriver(string driverShapeName)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            if (correctives != null)
            {
                foreach (var c in correctives)
                {
                    if (c.driverShape == driverShapeName && rawValues.ContainsKey(c.affectedShape))
                    {
                        float affectedRaw = rawValues[c.affectedShape];
                        float corrected = affectedRaw * ComputeCorrectiveFactor(c.affectedShape);
                        ApplyBlendshapeInternal(c.affectedShape, corrected);
                    }
                }
            }
            return;
        }
#endif

        if (correctiveByDriver == null) InitDictionary();
        if (correctiveByDriver.TryGetValue(driverShapeName, out var rules))
        {
            foreach (var c in rules)
            {
                if (rawValues.ContainsKey(c.affectedShape))
                {
                    float affectedRaw = rawValues[c.affectedShape];
                    float corrected = affectedRaw * ComputeCorrectiveFactor(c.affectedShape);
                    ApplyBlendshapeInternal(c.affectedShape, corrected);
                }
            }
        }
    }

    /// <summary>
    /// 外部接口：供你的 UDP 接收脚本调用，或供 Editor 界面调用
    /// </summary>
    /// <param name="shapeName">ARKit 标准名称 / 自定义名称</param>
    /// <param name="weight">权重 (0-100)</param>
    public void SetBlendshape(string shapeName, float weight)
    {
        // 0. 缓存原始值（修正系统需要用到）
        if (rawValues == null) rawValues = new Dictionary<string, float>();
        rawValues[shapeName] = weight;

        // 1. 先应用它原本的表情（带修正系数）
        float correctedWeight = weight * ComputeCorrectiveFactor(shapeName);
        ApplyBlendshapeInternal(shapeName, correctedWeight);

        // 2. [新增] 如果这个表情是某条修正规则的驱动源，重新修正受影响的表情
        ReapplyAffectedByDriver(shapeName);

        // 3. 检查是否有联动表情需要触发
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            if (shapeLinkages != null)
            {
                foreach (var link in shapeLinkages)
                {
                    if (link.sourceShape == shapeName)
                    {
                        float linkedRaw = weight * link.multiplier;
                        float linkedCorrected = linkedRaw * ComputeCorrectiveFactor(link.targetShape);
                        ApplyBlendshapeInternal(link.targetShape, linkedCorrected);
                    }
                }
            }
            return;
        }
#endif

        // 运行时使用字典快速触发联动
        if (linkageDictionary == null) InitDictionary();
        if (linkageDictionary.TryGetValue(shapeName, out var links))
        {
            foreach (var link in links)
            {
                float linkedRaw = weight * link.multiplier;
                float linkedCorrected = linkedRaw * ComputeCorrectiveFactor(link.targetShape);
                ApplyBlendshapeInternal(link.targetShape, linkedCorrected);
            }
        }
    }

    /// <summary>
    /// 底层实际应用 Blendshape 的方法 (包含安全锁逻辑)
    /// </summary>
    private void ApplyBlendshapeInternal(string shapeName, float weight)
    {
        // 触发安全锁：获取该表情的最大允许权重并截断
        float maxAllowed = GetMaxWeightLimit(shapeName);
        weight = Mathf.Clamp(weight, 0f, maxAllowed);

        if (targetMeshes == null) return;

        // 广播给所有网格
        foreach (var smr in targetMeshes)
        {
            if (smr == null || smr.sharedMesh == null) continue;

            int index = smr.sharedMesh.GetBlendShapeIndex(shapeName);
            if (index != -1)
            {
                smr.SetBlendShapeWeight(index, weight);
            }
        }
    }
}

// ==============================================================================
// 以下是自定义 Inspector 面板代码 (只在 Unity 编辑器中生效，打包时会自动忽略)
// ==============================================================================
#if UNITY_EDITOR
[CustomEditor(typeof(FaceBlendshapesManager))]
public class FaceBlendshapesManagerEditor : Editor
{
    private bool showBlendshapes = true;
    private List<string> allUniqueBlendshapes = new List<string>();

    private void OnEnable()
    {
        RefreshBlendshapeList();
    }

    // 收集所有网格的 Blendshapes 并集
    private void RefreshBlendshapeList()
    {
        FaceBlendshapesManager manager = (FaceBlendshapesManager)target;
        HashSet<string> uniqueNames = new HashSet<string>();

        if (manager.targetMeshes != null)
        {
            foreach (var smr in manager.targetMeshes)
            {
                if (smr != null && smr.sharedMesh != null)
                {
                    for (int i = 0; i < smr.sharedMesh.blendShapeCount; i++)
                    {
                        uniqueNames.Add(smr.sharedMesh.GetBlendShapeName(i));
                    }
                }
            }
        }

        allUniqueBlendshapes = new List<string>(uniqueNames);
        allUniqueBlendshapes.Sort();
    }

    public override void OnInspectorGUI()
    {
        // 绘制默认的面板 (targetMeshes, safeLimits, shapeLinkages, correctives 等)
        DrawDefaultInspector();

        FaceBlendshapesManager manager = (FaceBlendshapesManager)target;

        GUILayout.Space(15);

        // 标题栏与刷新按钮
        EditorGUILayout.BeginHorizontal();
        showBlendshapes = EditorGUILayout.Foldout(showBlendshapes, "联合 Blendshapes 测试面板", true, EditorStyles.foldoutHeader);
        if (GUILayout.Button("刷新列表", GUILayout.Width(80)))
        {
            RefreshBlendshapeList();
        }
        EditorGUILayout.EndHorizontal();

        if (showBlendshapes && allUniqueBlendshapes.Count > 0)
        {
            EditorGUILayout.HelpBox(
                "拖动滑动条会同时修改所有包含该名称的网格。\n" +
                "受【安全限制锁】和【修正混合】保护。",
                MessageType.Info);

            foreach (string shapeName in allUniqueBlendshapes)
            {
                // 获取当前该表情的值（从第一个含有该表情的网格身上获取）
                float currentValue = 0f;
                bool foundAny = false;
                foreach (var smr in manager.targetMeshes)
                {
                    if (smr != null && smr.sharedMesh != null)
                    {
                        int idx = smr.sharedMesh.GetBlendShapeIndex(shapeName);
                        if (idx != -1)
                        {
                            currentValue = smr.GetBlendShapeWeight(idx);
                            foundAny = true;
                            break;
                        }
                    }
                }

                if (!foundAny) continue;

                EditorGUI.BeginChangeCheck();
                float newValue = EditorGUILayout.Slider(shapeName, currentValue, 0f, 100f);
                if (EditorGUI.EndChangeCheck())
                {
                    // 记录 Undo 操作，支持 Ctrl+Z 撤销
                    foreach (var smr in manager.targetMeshes)
                    {
                        if (smr != null)
                        {
                            Undo.RecordObject(smr, $"Change Blendshape {shapeName}");
                        }
                    }

                    // 调用主脚本的接口设置权重，完整走安全锁 + 修正混合逻辑
                    manager.SetBlendshape(shapeName, newValue);
                }
            }
        }
        else if (showBlendshapes)
        {
            EditorGUILayout.LabelField("暂无数据，请先拖入带有 Blendshape 的 SkinnedMeshRenderer", EditorStyles.centeredGreyMiniLabel);
        }
    }
}
#endif