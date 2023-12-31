### Name generator

Model created with tensorflow library. It is trained with the names from "names.txt"
file which comes from [ssa.gov](https://www.ssa.gov/oact/babynames/). 

<details>
<summary>Sample few names from "names.txt" file</summary>

```
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
evelyn
abigail
...
```

</details>


<details>
<summary>Model output sample</summary>

```text
storey
jessian
mokamen
jamin
laiyah
lynkon
ema
rarlyn
noe
tenda
marrahlem
leyla
aviel
```

</details>

Cross-entropy loss got from the model:

```text
Model evaluation on train data
loss: 2.1383378505706787
Model evaluation on development data
loss: 2.1392717361450195
Model evaluation on test data
loss: 2.1262073516845703
```

It can be further improved by hyperparameters tuning

>[!IMPORTANT]
> This model is working with:
>   - tensorflow v2.14
>   - tensorflow-probability v0.22


