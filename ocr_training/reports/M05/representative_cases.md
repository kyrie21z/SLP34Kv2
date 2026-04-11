# M05 Representative Cases

说明：

- 以下样本来自 `evaluation/results/M05/case_inventory.csv`
- “子集标签”使用 `quality / layout / vocabulary_type` 与长度特征的组合描述
- “为什么归到该类”只给当前模块允许的统计性解释，不写强因果结论

## replace-dominant

### Case 1

- `image_id`: `5722`
- `model_tag`: `baseline`
- `gt`: `盐瑞龙1盐城YANRUILONGYANCHENG`
- `pred`: `浙钱江货00398杭州港ZHEQIANJIANGHUOHANGZHOUGANG`
- 子集标签：`hard / multi / OOV / long_21+`
- 为什么归到该类：`num_replace=16`，高于其他单类操作，属于典型 replace-dominant 失配

### Case 2

- `image_id`: `6526`
- `model_tag`: `length_head`
- `gt`: `赣九江货1531九江港GANJIUJIANGHUOJIUJIANGGANG`
- `pred`: `皖海1268蚌埠港WANHANGBONGBENGBUGANG`
- 子集标签：`middle / multi / OOV / long_21+`
- 为什么归到该类：`num_replace=18`，且 `num_insert=0`，替换是主要误差来源

### Case 3

- `image_id`: `6526`
- `model_tag`: `selective_eos_final`
- `gt`: `赣九江货1531九江港GANJIUJIANGHUOJIUJIANGGANG`
- `pred`: `皖海1268蚌埠港WANHANGBONGBENGBUGANG`
- 子集标签：`middle / multi / OOV / long_21+`
- 为什么归到该类：与 `length_head` 同样是高 replace 样本，说明 selective EOS 并未改变其主误差结构

## insert-dominant

### Case 1

- `image_id`: `1657`
- `model_tag`: `baseline`
- `gt`: `苏中川6118盐城港`
- `pred`: `皖中川6618淮安WANZHONGCHUANHUAN`
- 子集标签：`hard / multi / IV / 9-12`
- 为什么归到该类：`num_insert=16` 明显高于 `replace=5` 与 `delete=0`

### Case 2

- `image_id`: `1657`
- `model_tag`: `length_head`
- `gt`: `苏中川6118盐城港`
- `pred`: `皖中川6618盐城WANZHONGCHUANYANCHENG`
- 子集标签：`hard / multi / IV / 9-12`
- 为什么归到该类：`num_insert=20`，是典型过度扩写型错误

### Case 3

- `image_id`: `1657`
- `model_tag`: `selective_eos_final`
- `gt`: `苏中川6118盐城港`
- `pred`: `皖中川6618盐城WANZHONGCHUANYANCHENG`
- 子集标签：`hard / multi / IV / 9-12`
- 为什么归到该类：`num_insert=20`，且与 `length_head` 基本一致，说明该类样本不靠 EOS 线解决

## eos-related

### Case 1

- `image_id`: `5743`
- `model_tag`: `baseline`
- `gt`: `苏连云港1298连云港`
- `pred`: `苏连云港货1298连云港港SULIANYUNGANGHUOLIANYUNGANGGANG`
- 子集标签：`hard / multi / OOV / 9-12`
- 为什么归到该类：`eos_flag=eos_late`，并带有 `33` 个尾部插入

### Case 2

- `image_id`: `1478`
- `model_tag`: `length_head`
- `gt`: `浙萧山货13173杭州港`
- `pred`: `浙萧山货03173杭州港ZHEXIAOSHANHUOHANGZHOUGANG`
- 子集标签：`hard / multi / IV / 9-12`
- 为什么归到该类：`eos_flag=eos_late`，GT 结束后仍有大段冗余尾部

### Case 3

- `image_id`: `1435`
- `model_tag`: `selective_eos_final`
- `gt`: `浙富阳货00998杭州港`
- `pred`: `浙富阳货00826杭州港ZHEFUYANGHUOHANGZHOUGANG`
- 子集标签：`hard / multi / IV / 9-12`
- 为什么归到该类：`eos_flag=eos_late`，尾部冗余依然明显，说明 selective EOS 不能完全消除此类样本

## likely visual confusion

### Case 1

- `image_id`: `15`
- `model_tag`: `baseline`
- `gt`: `浙余杭货02039ZHEYUHANGHUO`
- `pred`: `浙余杭货02839ZHEYUHANGHUO`
- 子集标签：`easy / multi / IV / long_21+`
- 为什么归到该类：局部单点替换，`0 -> 8`，是典型高频数字视觉混淆

### Case 2

- `image_id`: `19`
- `model_tag`: `baseline`
- `gt`: `浙萧山货23653ZHEXIAOSHANHUO`
- `pred`: `浙萧山货23533ZHEXIAOSHANHUO`
- 子集标签：`easy / multi / IV / long_21+`
- 为什么归到该类：`6 -> 5`, `5 -> 3`，属于同类型数字替换，且对齐路径局部稳定

### Case 3

- `image_id`: `6526`
- `model_tag`: `length_head`
- `gt`: `赣九江货1531九江港GANJIUJIANGHUOJIUJIANGGANG`
- `pred`: `皖海1268蚌埠港WANHANGBONGBENGBUGANG`
- 子集标签：`middle / multi / OOV / long_21+`
- 为什么归到该类：大量替换集中在字符级映射上，`confusion_signature` 和位置分布都更像视觉/字形混淆主导

## likely layout/order issue

### Case 1

- `image_id`: `6477`
- `model_tag`: `baseline`
- `gt`: `苏顺发货2888淮安HUAIANSUSHUNFAHUO`
- `pred`: `湘张家货22888XIANGZHANGJIAJIEHUO`
- 子集标签：`middle / multi / OOV / long_21+`
- 为什么归到该类：中后段出现连续块状错位，位置签名覆盖 `middle + tail`

### Case 2

- `image_id`: `5964`
- `model_tag`: `length_head`
- `gt`: `宇洋889驻马店YUYANGZHUMADIAN`
- `pred`: `皖建888阜阳港WANJINQIANGFUYANGGANG`
- 子集标签：`middle / multi / OOV / long_21+`
- 为什么归到该类：`replace + insert + delete` 同时存在，且局部块状替换明显，不像单点字符视觉混淆

### Case 3

- `image_id`: `6501`
- `model_tag`: `length_head`
- `gt`: `豫周江河9188周口`
- `pred`: `豫周江河188周口YUZHOUJIANGHEZHOUKOU`
- 子集标签：`middle / multi / OOV / 9-12`
- 为什么归到该类：在主体基本保留后追加整段串，兼有顺序与尾部扩写特征

## likely language-prior-suspect

### Case 1

- `image_id`: `6168`
- `model_tag`: `baseline`
- `gt`: `浙萧山货23456杭州港`
- `pred`: `浙萧山货23456杭州港ZHEXIAOSHANHUOHANGZHOUGANG`
- 子集标签：`middle / multi / OOV / 9-12`
- 为什么归到该类：GT 已经完整命中，但后面追加了长串高频合法模式，更像语言先验/模式扩写

### Case 2

- `image_id`: `6318`
- `model_tag`: `baseline`
- `gt`: `皖明光货0045滁州港`
- `pred`: `皖明光货0045滁州港WANMINGGUANGHUOCHUZHOUGANG`
- 子集标签：`middle / multi / OOV / 9-12`
- 为什么归到该类：主体正确后继续扩写地名/拼音串，符合 `language_prior_suspect` 启发式

### Case 3

- `image_id`: `6168`
- `model_tag`: `selective_eos_final`
- `gt`: `浙萧山货23456杭州港`
- `pred`: `浙萧山货23456杭州港ZHEXIAOSHANHUOHANGZHOUGANG`
- 子集标签：`middle / multi / OOV / 9-12`
- 为什么归到该类：与 baseline 同型，说明这类扩写不是靠 selective EOS 就能自然收敛
