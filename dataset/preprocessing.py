import os
import json

captions = json.load(open("sft_caption.json"))
for caption in captions:
    caption["conversations"][1]["from"] = "gpt"
    if caption["conversations"][1]["value"] is None or "sorry" in caption["conversations"][1]["value"]:
        print(caption["conversations"][1]["value"])
        if os.path.exists(os.path.join("sft_img", caption["image"])):
            os.remove(os.path.join("sft_img", caption["image"]))
        captions.remove(caption)

# turn all key of "value" to "content"
for caption in captions:
    caption["conversations"][0]["content"] = caption["conversations"][0].pop("value")
    caption["conversations"][1]["content"] = caption["conversations"][1].pop("value")
# save into jsonl file
with open("sft_caption_updated.jsonl", "w") as f:
    for caption in captions:
        f.write(json.dumps(caption) + "\n")

