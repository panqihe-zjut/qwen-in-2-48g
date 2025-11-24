### Using Qwen-image with two 48g cards locally.
### (两张 48g 显卡推理qwen-image)

防止 48g OOM

qwenimage里的文件是从diffuser中down下来的. 
做法也非常简单,把text-encoder放到一张48g(>16g)显卡上, transformer和vae放到48g(>40g)显卡上. 
