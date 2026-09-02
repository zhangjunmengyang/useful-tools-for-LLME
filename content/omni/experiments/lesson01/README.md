# Lesson 01 execution

Remote persistent root:

```text
/mnt/tidal-alsh-share2/dataset/benchmark/_learn_omni
```

Locked inputs:

```text
code: jingyaogong/minimind-o@a10fa6c148ed274d66f96dc119689e93e01be823
data: jingyaogong/minimind-o_dataset@d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de
llm:  jingyaogong/minimind-3o-pytorch@9e22fb0e51852359ff51bd48d91ec3a345dbd75b
```

The first patch exposes dataset stochasticity as CLI arguments so the fixed
128-row overfit test can disable shuffle, scheduled sampling, random system
prompt insertion, and random empty-think removal.
