using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[System.Serializable]
public class EyeLookPair
{
    public string shape1;
    public string shape2;
}

// 1. 新增：眨眼速度档位配置类
[System.Serializable]
public class BlinkSpeedProfile
{
    public string profileName = "正常";
    [Tooltip("完成一次闭眼+睁眼的总耗时(秒)")]
    public float duration = 0.25f;
    [Tooltip("被抽中的概率权重 (值越大越容易触发)")]
    public float weight = 10f;
}

public class AutoBlinkAndLook : MonoBehaviour
{
    [Header("--- 核心依赖 ---")]
    public FaceBlendshapesManager blendshapesManager;

    [Header("--- 眨眼设置 (Blink) ---")]
    public bool enableAutoBlink = true;
    public string leftBlinkName = "eyeBlinkLeft";
    public string rightBlinkName = "eyeBlinkRight";
    public float minBlinkInterval = 2.0f;
    public float maxBlinkInterval = 6.0f;
    // 原有的 blinkDuration 可以删掉或者留作缺省值
    [Range(0f, 1f)] public float doubleBlinkChance = 0.2f;

    // 2. 新增：配置池变量
    [Header("--- 眨眼速率多档位池 (Blink Speed Pool) ---")]
    [Tooltip("配置不同的眨眼速度及其触发概率。默认配置了快、中、慢三种。")]
    public List<BlinkSpeedProfile> blinkSpeedPool = new List<BlinkSpeedProfile>
    {
        new BlinkSpeedProfile { profileName = "正常", duration = 0.25f, weight = 70f },
        new BlinkSpeedProfile { profileName = "慢速", duration = 0.45f, weight = 20f },
        new BlinkSpeedProfile { profileName = "极慢", duration = 0.70f, weight = 10f }
    };

    [Header("--- 眨眼速率曲线 (Blink Speed Curve) ---")]
    [Tooltip("在面板上绘制曲线。X轴0~1代表闭眼时间进度，Y轴0~1代表闭合程度。")]
    public AnimationCurve blinkCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);

    [Header("--- 眉眼连带肌肉 (Brow Linkage) ---")]
    public bool enableBrowLinkage = true;
    public string browDownLeftName = "browDownLeft";
    public string browDownRightName = "browDownRight";
    [Range(0f, 1f)] public float browLinkageChance = 0.5f;
    [Range(0f, 100f)] public float minBrowWeight = 20f;
    [Range(0f, 100f)] public float maxBrowWeight = 100f;

    [Header("--- 半眯眼状态 (ジト目) ---")]
    public bool enableJitoMe = true;
    public string jitoMeShapeName = "ジト目";
    public float jitoMeIntervalMin = 8.0f;
    public float jitoMeIntervalMax = 15.0f;
    public float jitoMeHoldTimeMin = 2.0f;
    public float jitoMeHoldTimeMax = 5.0f;
    [Range(0f, 100f)] public float jitoMeTargetWeight = 40f;
    public float jitoMeTransitionTime = 0.5f;

    private float currentJitoMeWeight = 0f;

    [Header("--- 眼神随机转动 (Saccades) ---")]
    public bool enableAutoLook = true;
    public float lookTransitionDuration = 0.08f;
    public float minLookHoldTime = 0.5f;
    public float maxLookHoldTime = 2.5f;
    public float maxLookWeight = 45f;
    public List<EyeLookPair> lookPairs = new List<EyeLookPair>
    {
        new EyeLookPair { shape1 = "eyeLookOutLeft", shape2 = "eyeLookInRight" },
        new EyeLookPair { shape1 = "eyeLookInLeft", shape2 = "eyeLookOutRight" },
        new EyeLookPair { shape1 = "eyeLookUpLeft", shape2 = "eyeLookUpRight" },
        new EyeLookPair { shape1 = "eyeLookDownLeft", shape2 = "eyeLookDownRight" }
    };

    private EyeLookPair currentLookPair = null;
    private float currentLookWeight = 0f;

    void Start()
    {
        if (blendshapesManager == null)
            blendshapesManager = GetComponent<FaceBlendshapesManager>();

        StartCoroutine(BlinkRoutine());
        StartCoroutine(LookRoutine());
        StartCoroutine(JitoMeRoutine());
    }

    // 3. 新增：权重随机算法函数
    private float GetRandomBlinkDuration()
    {
        if (blinkSpeedPool == null || blinkSpeedPool.Count == 0) return 0.25f;

        float totalWeight = 0f;
        foreach (var profile in blinkSpeedPool) totalWeight += profile.weight;

        float randomVal = Random.Range(0, totalWeight);
        float cumulativeWeight = 0f;

        foreach (var profile in blinkSpeedPool)
        {
            cumulativeWeight += profile.weight;
            if (randomVal <= cumulativeWeight)
            {
                return profile.duration;
            }
        }
        return blinkSpeedPool[blinkSpeedPool.Count - 1].duration;
    }

    // ================= 眨眼与眉毛联动逻辑 (已更新) =================
    IEnumerator BlinkRoutine()
    {
        while (true)
        {
            float waitTime = Random.Range(minBlinkInterval, maxBlinkInterval);
            yield return new WaitForSeconds(waitTime);

            if (enableAutoBlink && blendshapesManager != null)
            {
                yield return StartCoroutine(DoBlink());

                if (Random.value < doubleBlinkChance)
                {
                    yield return new WaitForSeconds(0.05f);
                    yield return StartCoroutine(DoBlink());
                }
            }
        }
    }

    IEnumerator DoBlink()
    {
        // --- 核心修改点：动态获取本次时长 ---
        float currentBlinkDuration = GetRandomBlinkDuration();
        float halfDuration = currentBlinkDuration / 2f;
        float elapsed = 0f;

        bool triggerBrow = enableBrowLinkage && (Random.value < browLinkageChance);
        float targetBrowWeight = triggerBrow ? Random.Range(minBrowWeight, maxBrowWeight) : 0f;
        float maxBlinkWeight = 100f - currentJitoMeWeight;

        // --- 闭眼过程 ---
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            float curveValue = blinkCurve.Evaluate(t);
            float currentBlink = curveValue * maxBlinkWeight;
            blendshapesManager.SetBlendshape(leftBlinkName, currentBlink);
            blendshapesManager.SetBlendshape(rightBlinkName, currentBlink);

            if (triggerBrow)
            {
                float currentBrow = curveValue * targetBrowWeight;
                blendshapesManager.SetBlendshape(browDownLeftName, currentBrow);
                blendshapesManager.SetBlendshape(browDownRightName, currentBrow);
            }
            yield return null;
        }

        // --- 睁眼过程 ---
        elapsed = 0f;
        while (elapsed < halfDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / halfDuration;
            float curveValue = blinkCurve.Evaluate(1f - t);
            float currentBlink = curveValue * maxBlinkWeight;
            blendshapesManager.SetBlendshape(leftBlinkName, currentBlink);
            blendshapesManager.SetBlendshape(rightBlinkName, currentBlink);

            if (triggerBrow)
            {
                float currentBrow = curveValue * targetBrowWeight;
                blendshapesManager.SetBlendshape(browDownLeftName, currentBrow);
                blendshapesManager.SetBlendshape(browDownRightName, currentBrow);
            }
            yield return null;
        }

        // 强制归零
        blendshapesManager.SetBlendshape(leftBlinkName, 0f);
        blendshapesManager.SetBlendshape(rightBlinkName, 0f);
        if (triggerBrow)
        {
            blendshapesManager.SetBlendshape(browDownLeftName, 0f);
            blendshapesManager.SetBlendshape(browDownRightName, 0f);
        }
    }

    // ================= 其余逻辑（半眯眼、转头等）保持不变 =================
    IEnumerator JitoMeRoutine()
    {
        while (true)
        {
            float waitTime = Random.Range(jitoMeIntervalMin, jitoMeIntervalMax);
            yield return new WaitForSeconds(waitTime);

            if (enableJitoMe && blendshapesManager != null)
            {
                float elapsed = 0f;
                while (elapsed < jitoMeTransitionTime)
                {
                    elapsed += Time.deltaTime;
                    currentJitoMeWeight = Mathf.Lerp(0f, jitoMeTargetWeight, elapsed / jitoMeTransitionTime);
                    blendshapesManager.SetBlendshape(jitoMeShapeName, currentJitoMeWeight);
                    yield return null;
                }
                currentJitoMeWeight = jitoMeTargetWeight;
                blendshapesManager.SetBlendshape(jitoMeShapeName, currentJitoMeWeight);

                float holdTime = Random.Range(jitoMeHoldTimeMin, jitoMeHoldTimeMax);
                yield return new WaitForSeconds(holdTime);

                elapsed = 0f;
                while (elapsed < jitoMeTransitionTime)
                {
                    elapsed += Time.deltaTime;
                    currentJitoMeWeight = Mathf.Lerp(jitoMeTargetWeight, 0f, elapsed / jitoMeTransitionTime);
                    blendshapesManager.SetBlendshape(jitoMeShapeName, currentJitoMeWeight);
                    yield return null;
                }
                currentJitoMeWeight = 0f;
                blendshapesManager.SetBlendshape(jitoMeShapeName, currentJitoMeWeight);
            }
        }
    }

    IEnumerator LookRoutine()
    {
        while (true)
        {
            float holdTime = Random.Range(minLookHoldTime, maxLookHoldTime);
            yield return new WaitForSeconds(holdTime);

            if (enableAutoLook && blendshapesManager != null && lookPairs.Count > 0)
            {
                bool returnToCenter = (Random.value < 0.3f);
                EyeLookPair nextPair = null;
                float targetWeight = 0f;

                if (!returnToCenter)
                {
                    nextPair = lookPairs[Random.Range(0, lookPairs.Count)];
                    targetWeight = Random.Range(15f, maxLookWeight);
                }

                yield return StartCoroutine(TransitionLook(nextPair, targetWeight));
            }
        }
    }

    IEnumerator TransitionLook(EyeLookPair nextPair, float targetWeight)
    {
        float elapsed = 0f;
        EyeLookPair prevPair = currentLookPair;
        float startWeight = currentLookWeight;

        while (elapsed < lookTransitionDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / lookTransitionDuration;
            t = Mathf.SmoothStep(0f, 1f, t);

            if (prevPair != null && prevPair != nextPair)
            {
                float fadeOutWeight = Mathf.Lerp(startWeight, 0f, t);
                blendshapesManager.SetBlendshape(prevPair.shape1, fadeOutWeight);
                blendshapesManager.SetBlendshape(prevPair.shape2, fadeOutWeight);
            }

            if (nextPair != null)
            {
                float currentStart = (prevPair == nextPair) ? startWeight : 0f;
                float fadeInWeight = Mathf.Lerp(currentStart, targetWeight, t);
                blendshapesManager.SetBlendshape(nextPair.shape1, fadeInWeight);
                blendshapesManager.SetBlendshape(nextPair.shape2, fadeInWeight);
            }

            yield return null;
        }

        if (prevPair != null && prevPair != nextPair)
        {
            blendshapesManager.SetBlendshape(prevPair.shape1, 0f);
            blendshapesManager.SetBlendshape(prevPair.shape2, 0f);
        }
        if (nextPair != null)
        {
            blendshapesManager.SetBlendshape(nextPair.shape1, targetWeight);
            blendshapesManager.SetBlendshape(nextPair.shape2, targetWeight);
        }

        currentLookPair = nextPair;
        currentLookWeight = targetWeight;
    }
}