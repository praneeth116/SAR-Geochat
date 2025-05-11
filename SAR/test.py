import ast
import re
A = "{\"fishing_vessels\": \"\", \"non_fishing_vessels\": \"\", \"non_vessels\": \"{<32><94>}{<169><314>}{<348><321>}\"}"


import argparse
import torch
import os
import json
from PIL import Image
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

def eval_single_input(args):
    # Disable torch initialization for efficiency
    disable_torch_init()

    # Load the model and related components
    model_path = os.path.expanduser(args.model_path)
    model_name ="geochat"  # Extract model name from path
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Prepare the question
    question = args.question
    if args.question_type == "ref":
        question = f"[refer] Give me the location of <p> {question} </p>"
    else:
        question = f"[grounding]{question}"

    if model.config.mm_use_im_start_end:
        question = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    # Prepare conversation prompt
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # Process the input image
    image = Image.open(args.image_path)
    image_tensor = image_processor.preprocess(
        images=image, 
        crop_size={"height": 504, "width": 504},
        size={"shortest_edge": 504}, 
        return_tensors="pt"
    )["pixel_values"].half().cuda()

    # Define stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    

    # Generate the output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, 
            images=image_tensor, 
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=75,
            length_penalty=2.0,
            
        )

    # Decode the output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    output = outputs[0].strip()
    if output.endswith(stop_str):
        output = output[:-len(stop_str)].strip()


    # Print the result
    print(json.dumps({
        "question": args.question,
        "image_path": args.image_path,
        "answer": output
    }, indent=2))
    d = ast.literal_eval(output)
    result=[]
    count=0
    for i in d.keys():
        input_string  = d[i]
        #result[count].append(i)
        # Use regular expressions to find all pairs of numbers
        pairs = re.findall(r'<(\d+)><(\d+)>', input_string)
        # Convert the pairs to a list of lists with integers
        list_of_pairs=list([int(x), int(y)] for x, y in pairs)
        result.append( list_of_pairs)
        
        count+=1
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/cvpr_ug_4/GeoChat/merged_model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-path", type=str, default="/home/cvpr_ug_4/SAR/single_channel_rgb/fe6a8d80fb5ebb8ev_18614_22700.png")
    parser.add_argument("--question", type=str, default="Detect and list the centers of all fishing vessels, non-fishing vessels, and non-vessels")
    parser.add_argument("--question-type", type=str, choices=["ref", "grounding"], default="grounding")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_single_input(args)
