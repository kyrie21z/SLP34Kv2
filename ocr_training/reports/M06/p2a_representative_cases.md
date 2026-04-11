# P2A Representative Cases

## pilot_focus_105
### m06_sub_train
#### Improved
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 5456 | 皖通祥1号皖蚌埠港 | 皖淮南货1155 | 皖塘塘619蚌埠港 | 8 | 5 | easy | single | OOV |
| 6621 | 浙杭州货10001 | 浙诸暨货1000 | 浙杭州货10001 | 3 | 0 | middle | single | OOV |
| 1899 | 豫信货0716 | 皖临泉货0186 | 皖临泉货0716 | 5 | 3 | hard | single | IV |
| 6660 | 兴中油258XINGZHONGYOU | 姑塘1619GUTANGCAOJI | 姑塘1619GUTANGZHOU | 16 | 14 | middle | single | OOV |
| 1829 | 浙绍兴货0616ZHESHAOXINGHUO | 浙绍兴货0618ZHESHAOXINGHUO | 浙绍兴货0616ZHESHAOXINGHUO | 1 | 0 | hard | single | IV |

#### Degraded
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 5802 | 浙富阳货00820 | 皖亳州货0820 | 皖蚌埠货0220WANBENGBUHUO | 4 | 17 | hard | single | OOV |
| 4224 | 浙富阳货01009ZHEFUYANGHUO | 浙富阳货01095ZHEFUYANGHUO | 浙富阳货01009 | 2 | 12 | middle | single | IV |
| 1896 | 菏泽港 | 浙庄港 | 浙诸暨货0077 | 2 | 8 | hard | single | IV |
| 4478 | 皖太运238阜阳港WANTAIYUNFUYANGGANG | 皖太运288阜阳港WANTAIYUNFUYANGGANG | 皖太运288阜阳WANTAIYUNFUYANG | 1 | 6 | middle | single | IV |
| 5801 | 浙富阳货00820 | 浙海盐货0220 | 皖海盛002 | 4 | 6 | hard | single | OOV |

### m06_joint_train
#### Improved
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 5456 | 皖通祥1号皖蚌埠港 | 皖淮南货1155 | 皖塘塘619蚌埠港 | 8 | 5 | easy | single | OOV |
| 6621 | 浙杭州货10001 | 浙诸暨货1000 | 浙杭州货10001 | 3 | 0 | middle | single | OOV |
| 6844 | 豫周江河7899 | 皖庐顺7899 | 豫周江河789 | 4 | 1 | middle | single | OOV |
| 1899 | 豫信货0716 | 皖临泉货0186 | 皖创业0716 | 5 | 3 | hard | single | IV |
| 6660 | 兴中油258XINGZHONGYOU | 姑塘1619GUTANGCAOJI | 姑塘1619GUTANGZHOU | 16 | 14 | middle | single | OOV |

#### Degraded
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 5802 | 浙富阳货00820 | 皖亳州货0820 | 皖亳州货0220WANBOZHOUHUO | 4 | 17 | hard | single | OOV |
| 4224 | 浙富阳货01009ZHEFUYANGHUO | 浙富阳货01095ZHEFUYANGHUO | 浙富阳货01009 | 2 | 12 | middle | single | IV |
| 1896 | 菏泽港 | 浙庄港 | 浙诸暨货0077绍兴港 | 2 | 10 | hard | single | IV |
| 4478 | 皖太运238阜阳港WANTAIYUNFUYANGGANG | 皖太运288阜阳港WANTAIYUNFUYANGGANG | 皖太运288阜阳WANTAIYUNFUYANG | 1 | 6 | middle | single | IV |
| 1773 | 鲁济宁货1628 | 鲁济宁货0728 | 鲁济宁货0723 | 2 | 3 | hard | single | IV |

## full_unified_6884
### m06_sub_train
#### Improved
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1435 | 浙富阳货00998杭州港 | 浙富阳货00826杭州港ZHEFUYANGHUOHANGZHOUGANG | 浙富阳货00878杭州港 | 27 | 2 | hard | multi | IV |
| 5592 | 浙上虞货0257绍兴港ZHESHANGYUHUOSHAOXINGGANG | 浙上虞货0257绍兴港 | 浙上虞货0257绍兴港ZHESHANGYUHUOSHAOXINGGANG | 25 | 0 | hard | multi | OOV |
| 5680 | 浙长兴货6605湖州港ZHECHANGXINGHUOHUZHOUGANG | 浙长兴货6605湖州港 | 浙长兴货6605湖州港ZHECHANGXINGHUOHUZHOUGANG | 25 | 0 | hard | multi | OOV |
| 6441 | 苏盐城货062698盐城港 | 苏盐城货062898盐城港SUYANCHENGHUOYANCHENGGANG | 苏盐城货062898盐城港 | 26 | 1 | middle | multi | OOV |
| 5601 | 浙余杭货02907杭州港ZHEYUHANGHUOHANGZHOUGANG | 浙余杭货02907杭州港 | 浙余杭货02907杭州港ZHEYUHANGHUOHANGZHOUGANG | 24 | 0 | hard | multi | OOV |

#### Degraded
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1558 | 浙越城货0078绍兴港 | 浙越城货0076绍兴港 | 浙越城货0078绍兴港ZHEYUECHENGHUOSHAOXINGGANG | 1 | 26 | hard | multi | IV |
| 1458 | 浙桐乡货02809 | 浙桐乡货01689 | 湘张家界货1189XIANGZHANGJIAJIEHUO | 3 | 26 | hard | multi | IV |
| 5688 | 皖东南1688合肥 | 皖兴隆1888蚌埠港 | 豫远翔1088周口港YUYUANXIANGZHOUKOUGANG | 6 | 29 | hard | multi | OOV |
| 1613 | 皖固镇货0781蚌埠港 | 皖固镇货0781蚌埠港 | 皖固镇货0781蚌埠港WANGUZHENHUOBENGBUGANG | 0 | 22 | hard | multi | IV |
| 5691 | 皖创业0156亳州港 | 皖创业0156亳州港 | 皖创业0156亳州港WANCHUANGYEBOZHOUGANG | 0 | 21 | hard | multi | OOV |

### m06_joint_train
#### Improved
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1435 | 浙富阳货00998杭州港 | 浙富阳货00826杭州港ZHEFUYANGHUOHANGZHOUGANG | 浙富阳货00878杭州港 | 27 | 2 | hard | multi | IV |
| 5592 | 浙上虞货0257绍兴港ZHESHANGYUHUOSHAOXINGGANG | 浙上虞货0257绍兴港 | 浙上虞货0257绍兴港ZHESHANGYUHUOSHAOXINGGANG | 25 | 0 | hard | multi | OOV |
| 5680 | 浙长兴货6605湖州港ZHECHANGXINGHUOHUZHOUGANG | 浙长兴货6605湖州港 | 浙长兴货6605湖州港ZHECHANGXINGHUOHUZHOUGANG | 25 | 0 | hard | multi | OOV |
| 5601 | 浙余杭货02907杭州港ZHEYUHANGHUOHANGZHOUGANG | 浙余杭货02907杭州港 | 浙余杭货02907杭州港ZHEYUHANGHUOHANGZHOUGANG | 24 | 0 | hard | multi | OOV |
| 1657 | 苏中川6118盐城港 | 皖中川6618盐城WANZHONGCHUANYANCHENG | 苏中川6618盐城港 | 23 | 1 | hard | multi | IV |

#### Degraded
| image_id | gt | raw pred | new pred | raw edits | new edits | quality | layout | vocab |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1558 | 浙越城货0078绍兴港 | 浙越城货0076绍兴港 | 浙越城货0078绍兴港ZHEYUECHENGHUOSHAOXINGGANG | 1 | 26 | hard | multi | IV |
| 1458 | 浙桐乡货02809 | 浙桐乡货01689 | 湘张家界货1989XIANGZHANGJIAJIEHUO | 3 | 26 | hard | multi | IV |
| 5688 | 皖东南1688合肥 | 皖兴隆1888蚌埠港 | 豫远翔1088周口港YUYUANXIANGZHOUKOUGANG | 6 | 29 | hard | multi | OOV |
| 5691 | 皖创业0156亳州港 | 皖创业0156亳州港 | 皖创业0156亳州港WANCHUANGYEBOZHOUGANG | 0 | 21 | hard | multi | OOV |
| 3215 | 皖永盛188亳州WANYONGSHENGBOZHOU | 皖永盛188亳州WANYONGSHENGBOZHOU | 皖永盛188亳州 | 0 | 18 | middle | multi | IV |
