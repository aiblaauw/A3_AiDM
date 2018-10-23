# A3_AiDM

find pairs of users where jsim(u1, u2) > 0.5
- Jaccard similarity = intersect / union
- Using Locality sensitive hashing algorithm with MinHashing
- Input user_movie.npy data
- Also read random seed number from command line   
- Output to textfile; list of records in the form u1,u2

**Usage**  

```
Using "user_movie.npy"

python main.py {RANDOMSEED} {FILEPATH} 

results.txt
{USER1,USER2}
```
