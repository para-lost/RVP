import datetime
import math
import os
import pathlib
from functools import partial
import warnings
import traceback
import re
import openai
import backoff
import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from configs import config
from utils import seed_everything
import datasets
import json
import ast
# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return

# After recursive call, check if the results are correct. If not, regenerate.
@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def check_code(orig_code, result, decomposed_code, question, query, codex, prompt_file='./prompts/correctness_check_chatapi.prompt'):
    
    with open(prompt_file) as f:
        base_prompt = f.read().strip()
    # Insert the instance into the base_prompt
    extended_prompt = [base_prompt.replace("INSERT_QUERY_HERE", query).replace("INSERT_CODE_HERE", orig_code).replace('INSERT_SUBQUERY_HERE', question).replace('INSERT_RESULT_HERE', result).replace('INSERT_RECURSIVE_CODE_HERE', decomposed_code)]
    
    responses = [openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.codex.temperature,
            max_tokens=config.codex.max_tokens,
            top_p = 1.,
            frequency_penalty=0,
            presence_penalty=0,
#                 best_of=config.codex.best_of,
            stop=["\n\n"],
            )
                for prompt in extended_prompt]
    result = [r['choices'][0]['message']['content'].replace("execute_command(image)", "execute_command(image, my_fig, time_wait_between_lines, syntax)") for r in responses]
    
    result = result[0]
        
    # result = codex.check_correctness(orig_code, decomposed_code, result, question, query)
    print(result)

    if 'Yes' == result:
        return True
    else:
        return False

def extract_start_index(query_string):
    match = re.search(r'start_index=\(*\)', query_string)
    if match:
        return int(match.group(1))
    else:
        return 0

def recursive_run_program_for_video2(parameters, queues_in_, input_type_, codex, query, base_prompt, non_recursive_ans=None, retrying=False, depth=0):
    if depth > 10:
        print("Recursion depth exceeded!")
        return None, None

    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool,  recursive_query
    from video_segment import VideoSegment, select_answer

    global queue_results

    code, sample_id, video, possible_answers, query = parameters
    if isinstance(code, list):
        code = code[0]
    if 'Failed!!!' in code:
        print("Directly Using the fixed code")
        new_code = "["  # Placeholder error code
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
        if non_recursive_ans != None:
            result = non_recursive_ans
        return result, code
    print(code)
    pattern = re.compile(r'recursive_query\("(.*?)"(?:,|\))')
    segments = code.split('\n')

    # We will build the final code incrementally
    final_code = ""
    orig_code = ""
    tot = ""
    for segment in segments:
        orig_code += segment + "\n"

    for segment in segments:
        if pattern.search(segment):
            question = pattern.search(segment).group(1)
            start_index = extract_start_index(segment)
            print("start index is:" , start_index)
            video_segment = VideoSegment(video)
            video_recursive = video_segment.trim(start_index)
            new_code = codex(prompt=[question], base_prompt=base_prompt)  # Generate code for the new query
            if isinstance(new_code, list):
                new_code = new_code[0]
            if question in new_code:
                segment = segment.replace("recursive", "simple")
                final_code += segment + "\n"
            else:
                result, decomposed_code = recursive_run_program_for_video((new_code, sample_id, video_recursive, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt,retrying=False, depth=depth+1)                    
                if '=' in segment:
                    segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                else:
                    pattern2 = r'recursive_query\(".*?"\)'
                    segment_replacement = re.sub(pattern2, repr(result), segment)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
        else:
            final_code += segment + "\n"

    # Now, we have the final code without any recursive_query calls
    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query):\n' \
                  f'    # Answer is:'
    code = code_header + final_code
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        result = globals()[f'execute_command_{sample_id}'](
            video, possible_answers, query,
            partial(ImagePatch, queues=[queues_in_, queue_results]),
            partial(VideoSegment, queues=[queues_in_, queue_results]),
            partial(llm_query, queues=[queues_in_, queue_results]),
            bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query
        )
    except Exception as e:
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        new_code = "["  # Placeholder error code
        if non_recursive_ans != None:
            result = non_recursive_ans
        else:
            result = run_program((new_code, sample_id, video, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
    # Cleanup
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    code = orig_code + tot 
    return result, code




def get_fixed_code(question):
    prompt = "def execute_command(video):\n    question = \""+question+"\"\n    video_segment = VideoSegment(video)\n    first_frame = video_segment.frame_from_index(0)\n    mid_frame = video_segment.frame_from_index(video_segment.num_frames//2)\n    last_frame = video_segment.frame_from_index(-1)\n    first_frame_query_answer = first_frame.simple_query(\"Caption: \")\n    mid_frame_query_answer = mid_frame.simple_query(\"Caption: \")\n    last_frame_query_answer = last_frame.simple_query(\"Caption: \")\n    info = {\n        \"First frame of the video\": first_frame_query_answer, \n        \"Middle frame of the video\": mid_frame_query_answer, \n        \"Last frame of the video\": last_frame_query_answer, \n    }\n    answer = get_text_answer(info, question)\n    return answer"
    return prompt



def recursive_run_program_for_video(parameters, queues_in_, input_type_, codex, query, base_prompt, base_prompt_nonrecursive, memory_bank={}, non_recursive_ans=None, retrying=False, depth=0):
    if depth > 10:
        print("Recursion depth exceeded!")
        return None, None

    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool,  recursive_query, get_text_answer
    from video_segment import VideoSegment, select_answer

    global queue_results
    
    code, sample_id, video, possible_answers, query = parameters
    orig_memory_bank = memory_bank
    print(memory_bank)
    if isinstance(code, list):
        code = code[0]
    if 'Failed!!!' in code:
        print("Directly Using the fixed code")
        new_code = "["  # Placeholder error code
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
        if non_recursive_ans != None:
            result = non_recursive_ans
        return result, code
    pattern = re.compile(r'recursive_query\("(.*?)"(?:,|\))')
    segments = code.split('\n')

    # We will build the final code incrementally
    final_code = ""
    orig_code = ""
    tot = ""
    for segment in segments:
        orig_code += segment + "\n"
    base_prompt = base_prompt.split("# INSERT_QUERY_HERE")[0] + "\n# " + query + "\n" + orig_code + "\n# INSERT_QUERY_HERE"
    
    for segment in segments:
        if pattern.search(segment):
            question = pattern.search(segment).group(1)
            start_index = extract_start_index(segment)
            # print("start index is:" , start_index)
            video_segment = VideoSegment(video)
            video_recursive = video_segment.trim(start_index) 
            # Change back to base_prompt=base_prompt for the original version
            new_code = codex(prompt=[question], memory_bank=memory_bank, base_prompt=base_prompt_nonrecursive)  # Generate code for the new query
            if isinstance(new_code, list):
                new_code = new_code[0]
            if question in new_code and 'recursive_query' in new_code:
                new_code = get_fixed_code(question)
                result, decomposed_code = recursive_run_program_for_video((new_code, sample_id, video_recursive, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt, base_prompt_nonrecursive, memory_bank=memory_bank,retrying=False, depth=depth+1)
                # segment = segment.split('",')[0]+'")'
                # segment = segment.replace("recursive", "video_segment.frame_from_index(0).simple")
                memory_bank[question] = result
                segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
            else:
                result, decomposed_code = recursive_run_program_for_video((new_code, sample_id, video_recursive, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt, base_prompt_nonrecursive, memory_bank=memory_bank,retrying=False, depth=depth+1)                    
                memory_bank[question] = result
                print(result)
                if '=' in segment:
                    segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                else:
                    pattern2 = r'recursive_query\(".*?"\)'
                    segment_replacement = re.sub(pattern2, repr(result), segment)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
        else:
            final_code += segment + "\n"

    # Now, we have the final code without any recursive_query calls
    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query, memory_bank, get_text_answer):\n' \
                  f'    # Answer is:'
    if 'def execute_command' in final_code:
        final_code = 'def execute_command' + final_code.split('def execute_command')[1]
    code = code_header + final_code
    print(code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        result = globals()[f'execute_command_{sample_id}'](
            video, possible_answers, query,
            partial(ImagePatch, queues=[queues_in_, queue_results]),
            partial(VideoSegment, queues=[queues_in_, queue_results]),
            partial(llm_query, queues=[queues_in_, queue_results]),
            bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query, memory_bank, get_text_answer
        )
    except Exception as e:
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        new_code = "["  # Placeholder error code
        if non_recursive_ans != None:
            result = non_recursive_ans
        else:
            result = run_program((new_code, sample_id, video, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
    # Cleanup
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    code = str(memory_bank) + orig_code + tot 
    return result, code


def recursive_run_program(parameters, queues_in_, input_type_, codex, query, base_prompt, non_recursive_ans=None, retrying=False, depth=0):
    if depth > 10:
        print("Recursion depth exceeded!")
        return None, None

    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, recursive_query
    from video_segment import VideoSegment, select_answer

    global queue_results
    non_recursive_ans=None
    code, sample_id, image, possible_answers, query = parameters
    if isinstance(code, list):
        code = code[0]
    if 'Failed!!!' in code:
        print("Directly Using the fixed code")
        new_code = "["  # Placeholder error code
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
        if non_recursive_ans != None:
            result = non_recursive_ans
        return result, code
    print(code)
    pattern = re.compile(r'recursive_query\("(.*?)"\)')
    segments = code.split('\n')

    # We will build the final code incrementally
    final_code = ""
    orig_code = ""
    tot = ""
    for segment in segments:
        orig_code += segment + "\n"
    base_prompt = base_prompt.split("# INSERT_QUERY_HERE")[0] + "\n# " + query + "\n" + orig_code + "\n# INSERT_QUERY_HERE"
    for segment in segments:
        if pattern.search(segment):
            question = pattern.search(segment).group(1)
            new_code = codex(prompt=[question], base_prompt=base_prompt)  # Generate code for the new query
            if isinstance(new_code, list):
                new_code = new_code[0]
            if question in new_code and 'recursive_query' in new_code:
                # new_code = new_code.replace("recursive", "simple")
                # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
                # if '=' in segment:
                #     segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                # else:
                #     pattern2 = r'recursive_query\(".*?"\)'
                #     segment_replacement = re.sub(pattern2, repr(result), segment)
                segment = segment.replace("recursive", "simple")
                final_code += segment + "\n"
            else:
                result, decomposed_code = recursive_run_program((new_code, sample_id, image, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt,retrying=False, depth=depth+1)                    
                if '=' in segment:
                    segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                else:
                    pattern2 = r'recursive_query\(".*?"\)'
                    segment_replacement = re.sub(pattern2, repr(result), segment)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
        else:
            final_code += segment + "\n"

    # Now, we have the final code without any recursive_query calls
    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query):\n' \
                  f'    # Answer is:'
    code = code_header + final_code
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        result = globals()[f'execute_command_{sample_id}'](
            image, possible_answers, query,
            partial(ImagePatch, queues=[queues_in_, queue_results]),
            partial(VideoSegment, queues=[queues_in_, queue_results]),
            partial(llm_query, queues=[queues_in_, queue_results]),
            bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer,recursive_query
        )
    except Exception as e:
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        new_code = "["  # Placeholder error code
        if non_recursive_ans != None:
            result = non_recursive_ans
        else:
            result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
    # Cleanup
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    code = orig_code + tot 
    return result, code

def recursive_run_program_image(parameters, queues_in_, input_type_, codex, query, base_prompt, base_prompt_nonrecursive, example=None, non_recursive_ans=None, retrying=False, depth=0, use_retrieval=True):
    if depth > 10:
        print("Recursion depth exceeded!")
        return None, None

    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, recursive_query
    from video_segment import VideoSegment, select_answer
    import torch
    global queue_results
    start_id = 0
    result_name = ['result0','result1','result2','result3','result4']
    all_results = {}
    non_recursive_ans=None
    code, sample_id, image, possible_answers, query = parameters
    if isinstance(code, list):
        code = code[0]
    if 'Failed!!!' in code:
        print("Directly Using the fixed code")
        new_code = "["  # Placeholder error code
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
        if non_recursive_ans != None:
            result = non_recursive_ans
        return result, code
    print(code)
    pattern = re.compile(r'recursive_query\("(.*?)"\)')
    segments = code.split('\n')

    # We will build the final code incrementally
    final_code = ""
    orig_code = ""
    tot = ""
    for segment in segments:
        orig_code += segment + "\n"
    # base_prompt = base_prompt.split("# INSERT_QUERY_HERE")[0] + "\n# " + query + "\n" + orig_code + "\n# INSERT_QUERY_HERE"
    for segment in segments:
        if pattern.search(segment):
            question = pattern.search(segment).group(1)
            # if 'patch' in segment.split('=')[0] and 'ImagePatch' not in question:
            #     question = "Return an ImagePatch, " + question
            if use_retrieval:
                base_prompt_nonrecursive_query = get_base_prompt(base_prompt_nonrecursive, question, example[0], example[1], example[2])
            else: 
                base_prompt_nonrecursive_query = base_prompt_nonrecursive
            # new_code = codex(prompt=[question], base_prompt=base_prompt_nonrecursive_query)  # Generate code for the new query
            new_code = codex(prompt=[question], base_prompt=base_prompt_nonrecursive)
            new_code[0] = new_code[0].strip("```python").strip("```").strip()
            if isinstance(new_code, list):
                new_code = new_code[0]
            if question in new_code and 'recursive_query' in new_code:
                # new_code = new_code.replace("recursive", "simple")
                # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
                # if '=' in segment:
                #     segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                # else:
                #     pattern2 = r'recursive_query\(".*?"\)'
                #     segment_replacement = re.sub(pattern2, repr(result), segment)
                segment = segment.replace("recursive", "simple")
                final_code += segment + "\n"
            else:
                result, decomposed_code = recursive_run_program_image((new_code, sample_id, image, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt,base_prompt_nonrecursive, example, retrying=False, depth=depth+1)                    
                name = result_name[start_id]
                start_id += 1
                all_results[name] = result
                if '=' in segment:
                    # segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                    segment_replacement = segment.split('=')[0] + '= ' + f'all_results["{name}"]'
                else:
                    pattern2 = r'recursive_query\(".*?"\)'
                    segment_replacement = re.sub(pattern2, repr(result), segment)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
        else:
            final_code += segment + "\n"

    # Now, we have the final code without any recursive_query calls
    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query, torch, all_results):\n' \
                  f'    # Answer is:'
    code = code_header + final_code#.split('\ndef execute_command')[1] (used for davinci-003)
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        result = globals()[f'execute_command_{sample_id}'](
            image, possible_answers, query,
            partial(ImagePatch, queues=[queues_in_, queue_results]),
            partial(VideoSegment, queues=[queues_in_, queue_results]),
            partial(llm_query, queues=[queues_in_, queue_results]),
            bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer,recursive_query, torch, all_results
        )
    except Exception as e:
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        new_code = "["  # Placeholder error code
        if non_recursive_ans != None:
            result = non_recursive_ans
        else:
            result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
    # Cleanup
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    code = orig_code + tot 
    return result, code


def run_program_video(parameters, queues_in_, input_type_, non_recursive_ans=None, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, coerce_to_numeric, yesno_to_bool,  recursive_query, get_text_answer
    from video_segment import VideoSegment, select_answer

    global queue_results

    code, sample_id, video, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, coerce_to_numeric, yesno_to_bool,select_answer, recursive_query, get_text_answer):\n' \
                  f'    # Answer is:'
    code = code_header + code
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            print("Using fixed code:")
            print(code)
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            video, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query, get_text_answer)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, video, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code


def run_program(parameters, queues_in_, input_type_, non_recursive_ans=None, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, coerce_to_numeric, yesno_to_bool,  recursive_query, get_text_answer
    from video_segment import VideoSegment, select_answer

    global queue_results

    code, sample_id, image, possible_answers, query = parameters
    if input_type_ == 'images':
        input_type_ = 'image_list'
    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, coerce_to_numeric, yesno_to_bool,select_answer, recursive_query, get_text_answer):\n' \
                  f'    # Answer is:'
    code = code_header + code
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query, get_text_answer)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code

def recursive_run_program_for_multiimage(parameters, queues_in_, input_type_, codex, query, base_prompt, base_prompt_nonrecursive, non_recursive_ans=None, retrying=False, depth=0, already_nonrecursive=False):
    if depth > 10:
        print("Recursion depth exceeded!")
        return None, None

    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, recursive_query
    from video_segment import VideoSegment, select_answer

    global queue_results
    non_recursive_ans=None
    code, sample_id, image, possible_answers, query = parameters
    if isinstance(code, list):
        code = code[0]
    if 'Failed!!!' in code:
        print("Directly Using the fixed code")
        new_code = "["  # Placeholder error code
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
        if non_recursive_ans != None:
            result = non_recursive_ans
        return result, code
    print(code)
    pattern = re.compile(r'recursive_query\([^,]+,\s*"([^"]+)"\)')
    segments = code.split('\n')

    # We will build the final code incrementally
    final_code = ""
    orig_code = ""
    tot = ""
    for segment in segments:
        orig_code += segment + "\n"
    base_prompt = base_prompt.split("# INSERT_QUERY_HERE")[0] + "\n# " + query + "\n" + orig_code + "\n# INSERT_QUERY_HERE"
    for segment in segments:
        if pattern.search(segment):
            question = pattern.search(segment).group(1)
            new_code = codex(prompt=[question], base_prompt=base_prompt_nonrecursive)  # Generate code for the new query
            new_code[0] = new_code[0].strip("```python").strip("```").strip()
            if isinstance(new_code, list):
                new_code = new_code[0]
            if question in new_code and 'recursive_query' in new_code:
                # new_code = new_code.replace("recursive", "simple")
                # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
                # if '=' in segment:
                #     segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                # else:
                #     pattern2 = r'recursive_query\(".*?"\)'
                #     segment_replacement = re.sub(pattern2, repr(result), segment)
                
                # new_code = codex(prompt=[question], base_prompt=base_prompt_nonrecursive)
                # new_code[0] = new_code[0].strip("```python").strip("```").strip()
                # result, decomposed_code = recursive_run_program_for_multiimage((new_code, sample_id, image, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt, base_prompt_nonrecursive,retrying=False, depth=depth+1)                    
                # segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                # final_code += segment_replacement + "\n"
                # tot += "\n\n### Using simple_prompt:\n"
                # tot += decomposed_code  
                
                
                segment = segment.replace("recursive", "ImagePatch(image_list[0]).simple")
                segment = segment.replace("image_list, ", "")
                final_code += segment + "\n"
            else:
                result, decomposed_code = recursive_run_program_for_multiimage((new_code, sample_id, image, possible_answers, question), queues_in_, input_type_, codex, question, base_prompt, base_prompt_nonrecursive,retrying=False, depth=depth+1)                    
                if '=' in segment:
                    segment_replacement = segment.split('=')[0] + '= ' + repr(result)
                else:
                    pattern2 = r'recursive_query\(".*?"\)'
                    segment_replacement = re.sub(pattern2, repr(result), segment)
                final_code += segment_replacement + "\n"
                tot += "\n\n###\n"
                tot += decomposed_code  
        else:
            final_code += segment + "\n"

    # Now, we have the final code without any recursive_query calls
    code_header = f'def execute_command_{sample_id}(' \
                  f'image_list, possible_answers, query,' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer, recursive_query):\n' \
                  f'    # Answer is:'
    # Sometimes there is a '\n' in the front of the def execute_command code, which would cause an error
    print(final_code)
    if 'def execute_command' in final_code:
        final_code = 'def execute_command' + final_code.split('def execute_command')[1]
    code = code_header + final_code
    print("current code is" + code)
    try:
        exec(compile(code, 'Codex', 'exec'), globals())
        result = globals()[f'execute_command_{sample_id}'](
            image, possible_answers, query,
            partial(ImagePatch, queues=[queues_in_, queue_results]),
            partial(VideoSegment, queues=[queues_in_, queue_results]),
            partial(llm_query, queues=[queues_in_, queue_results]),
            bool_to_yesno, distance, best_image_match, direction_vector, calculate_angle, coerce_to_numeric, yesno_to_bool, select_answer,recursive_query
        )
    except Exception as e:
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # new_code = "["  # Placeholder error code
        # if non_recursive_ans != None:
        #     result = non_recursive_ans
        # else:
        # Default to execute non-recursive code:
        if not already_nonrecursive:
            new_code = codex(prompt=[query], base_prompt=base_prompt_nonrecursive)
            result = recursive_run_program_for_multiimage((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, codex, query, base_prompt, base_prompt_nonrecursive,retrying=False, depth=depth+1, already_nonrecursive=True)[0]                 
        else:
            new_code = "["
            result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_, retrying=True)[0]
    # Cleanup
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']
    code = orig_code + tot #+ "\n\nAfter Changed:\n" + final_code
    return result, code

def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]

def get_model_and_embedding(config):
    # Load the BERT-based model pre-trained on NLI and STSb datasets
    model = SentenceTransformer("all-mpnet-base-v2")
    
    with open(config.codex.prompt_pool, 'r') as file:
        data = json.load(file)
    with open(config.codex.prompt_nonrecursive_pool, 'r') as file:
        data_nonrecursive = json.load(file)
        
    # Extract questions and their IDs from the data
    ids = list(data.keys())
    questions = [item['question'] for item in data.values()]

    ids_nonrecursive = list(data_nonrecursive.keys())
    questions_nonrecursive = [item['question'] for item in data_nonrecursive.values()]
    
    questions_embeddings = model.encode(questions, convert_to_tensor=True)
    questions_nonrecursive_embeddings = model.encode(questions_nonrecursive, convert_to_tensor=True)

    return model, questions_embeddings, questions_nonrecursive_embeddings, data, data_nonrecursive

def get_base_prompt(base_prompt, query, model,questions_embeddings, data,top_n=8):
    # Embed the query and all questions
    ids = list(data.keys())
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities between query and each question
    similarities = []
    for q_id, question_embedding in zip(ids, questions_embeddings):
        cosine_similarity = util.pytorch_cos_sim(query_embedding, question_embedding)
        similarities.append((q_id, cosine_similarity.item()))

    # Sort by similarity
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Extract top N programs
    top_programs = [data[item[0]]['program'] for item in sorted_similarities[:top_n]]
    top_programs_str = ""
    
    for i, program in enumerate(top_programs):
        title = f"\n\nExample {i+1}"
        top_programs_str += title
        top_programs_str += program
    base_prompt = base_prompt.replace("# INSERT_EXAMPLES_HERE", top_programs_str)
    return base_prompt

def get_base_prompt_samefile(base_prompt, query, model,questions_embeddings, data, data_recursive_query_samples, tot_dict={}, top_n=3):
    # Embed the query and all questions
    ids = list(data.keys())
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarities between query and each question
    similarities = []
    for q_id, question_embedding in zip(ids, questions_embeddings):
        question_embedding = question_embedding.to(query_embedding.device)
        cosine_similarity = util.pytorch_cos_sim(query_embedding, question_embedding)
        similarities.append((q_id, cosine_similarity.item()))

    # Sort by similarity
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Extract top N programs
    top_programs = [data[item[0]]['program'] for item in sorted_similarities[:top_n]]
    top_programs_str = ""
    
    for i, program in enumerate(top_programs):
        title = f"\n\nExample {i+1}"
        top_programs_str += title
        top_programs_str += program
    top_recursive_programs_id = []
    if 'recursive_query' in top_programs_str:
        for item in sorted_similarities[:top_n]:
            if item[0] not in tot_dict.keys():
                tot_dict[item[0]] = 1
            else:
                tot_dict[item[0]] += 1
            for recursive_id in data[item[0]]["non_recursive_program_id"]:
                top_recursive_programs_id.append(recursive_id)
    if len(top_recursive_programs_id) > 0:
        top_programs_str += '\n\nRecursive Query Examples:'
        for recursive_id in top_recursive_programs_id:
            top_programs_str += "\n"
            top_programs_str += data_recursive_query_samples[recursive_id]["program"]    
    base_prompt = base_prompt.replace("# INSERT_EXAMPLES_HERE", top_programs_str)
    # print(base_prompt)
    return base_prompt, tot_dict



def main():
    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets.dataset import MyDataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)


    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]
    usecodellama = config.load_models.codellama                                     # Which pretrained models to load
    userecursive = config.use_recursive
    usedefault = config.use_default
    if not usecodellama:
        codex = partial(forward, model_name='codex', queues=[queues_in, queue_results_main])
    else:
        codex = partial(forward, model_name='codellama', queues=[queues_in, queue_results_main])
    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = MyDataset(**config.dataset)

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()
    # if usedefault:
    with open(config.codex.prompt) as f:
        base_prompt_nonrecursive = f.read().strip()
            
    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        # codes_all = [r.split("{}")[1].replace('select_answer(', 'select_answer(video, ') for r in results['code']]
        codes_all = [r for r in results['code']]
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    all_results = []
    all_answers = []
    all_codes = []
    all_ids = []
    all_querys = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    tot_dict = {}
    if config.save:
        results_dir = pathlib.Path(config['results_dir'])
        results_dir = results_dir / config.dataset.split
        results_dir.mkdir(parents=True, exist_ok=True)
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            existing_files = list(results_dir.glob('results_*.csv'))
            if len(existing_files) == 0:
                filename = 'results_0.csv'
            else:
                filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'

    if 'gqa' in config.dataset.split:
        example_model, example_questions_embeddings, example_questions_nonrecursive_embeddings,example_data, example_nonrecursive_data = get_model_and_embedding(config)                                            
    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            import csv
            n_batches = len(dataloader)
            
            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # Combine all querys and get Codex predictions for them
                # TODO compute Codex for next batch as current batch is being processed
                # non_recursive_ans = csv_data[i]["result"]
                if not config.use_cached_codex:
                    print(batch['query'])
                    # base_prompt_query, tot_dict = get_base_prompt_samefile(base_prompt, batch['query'][0], example_model, example_questions_embeddings, example_data, example_nonrecursive_data, tot_dict)
                    # codes = codex(prompt=batch['query'], base_prompt=base_prompt_query)
                    #  original (non-retrieval)
                    codes = codex(prompt=batch['query'], base_prompt=base_prompt)

                    codes[0] = codes[0].strip("```python").strip("```").strip()


                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache
                
                # Run the code
                if config.execute_code:
                    if not config.multiprocessing:
                        # Otherwise, we would create a new model for every process
                        results = []
                        if not userecursive:
                            if config.dataset.input_type == "image":
                                for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                    result = run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                                    results.append(result)
                            else:
                                for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                    result = run_program_video([c, sample_id, img, possible_answers, query], queues_in, input_type)
                                    results.append(result)
                            
                        else:
                            if usecodellama:
                                if config.dataset.input_type == "image":
                                    # for c, sample_id, img, possible_answers, query in \
                                    #         zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                    #     result = recursive_run_program([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt)
                                    #     results.append(result)
                                    for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        # base_prompt_query = get_base_prompt(base_prompt, query, example_model, example_questions_embeddings, example_data)
                                        base_prompt_query = base_prompt
                                        result = recursive_run_program_image([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt_query, base_prompt_nonrecursive, [example_model,example_questions_nonrecursive_embeddings, example_nonrecursive_data])
                                        results.append(result)
                                elif config.dataset.input_type == "images":
                                    for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        result = recursive_run_program_for_multiimage([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt, base_prompt_nonrecursive)
                                        results.append(result)
                                else:
                                    for c, sample_id, img, possible_answers, query in \
                                            zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        result = recursive_run_program_for_video([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt)
                                        results.append(result)
                            else:
                                if config.dataset.input_type == "image":
                                    for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        if not usedefault:
                                            base_prompt_query = get_base_prompt(base_prompt, query, example_model, example_questions_embeddings, example_data)
                                        # base_prompt_query, tot_dict = get_base_prompt_samefile(base_prompt, query, example_model, example_questions_embeddings, example_data, example_nonrecursive_data, tot_dict)
                                        else:
                                            base_prompt_query = base_prompt
                                        result = recursive_run_program_image([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt_query, base_prompt_nonrecursive, [example_model,example_questions_nonrecursive_embeddings, example_nonrecursive_data])
                                        results.append(result)
                                elif config.dataset.input_type == "images":
                                    for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        result = recursive_run_program_for_multiimage([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt, base_prompt_nonrecursive)
                                        results.append(result)
                                else:
                                    for c, sample_id, img, possible_answers, query in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                        result = recursive_run_program_for_video([c, sample_id, img, possible_answers, query], queues_in, input_type, codex, batch['query'], base_prompt, base_prompt_nonrecursive, memory_bank={})
                                        results.append(result)
                    else:
                        results = list(pool.imap(partial(
                            run_program, queues_in_=queues_in, input_type_=input_type),
                            zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])))
                else:
                    results = [(None, c) for c in codes]
                    warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                  "'execute_code' to False by default, because executing code generated by a language "
                                  "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                  "it.")

                all_results += [r[0] for r in results]
                all_codes += [r[1] for r in results]
                all_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['query_type']
                all_querys += batch['query']
                all_img_paths += [dataset.get_sample_path(idx) for idx in batch['index']]
                if i % config.log_every == 0:
                    try:
                        accuracy = datasets.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')
                    if config.save:
                        print('Saving results to', filename)
                        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths,
                                        all_possible_answers]).T
                        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
                        # make the result column a string
                        df['result'] = df['result'].apply(str)
                        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')
                        # torch.save([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths], results_dir/filename)

                        if config.wandb:
                            wandb.log({'accuracy': accuracy})
                            wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})

        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        accuracy = datasets.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save:
        print('Saving results to', filename)
        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths,
                        all_possible_answers]).T
        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
        # make the result column a string
        df['result'] = df['result'].apply(str)
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')
        # torch.save([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths], results_dir/filename)

        if config.wandb:
            wandb.log({'accuracy': accuracy})
            wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})
    print(tot_dict)


    finish_all_consumers()


if __name__ == '__main__':
    main()
