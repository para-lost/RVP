from __future__ import annotations

import torch
from typing import Union, Iterator

from image_patch import ImagePatch
from vision_processes import forward, config
import backoff
import openai

class VideoSegment:
    """A Python class containing a set of frames represented as ImagePatch objects, as well as relevant information.
    Attributes
    ----------
    video : torch.Tensor
        A tensor of the original video.
    start : int
        An int describing the starting frame in this video segment with respect to the original video.
    end : int
        An int describing the ending frame in this video segment with respect to the original video.
    num_frames->int
        An int containing the number of frames in the video segment.

    Methods
    -------
    frame_iterator->Iterator[ImagePatch]
    trim(start, end)->VideoSegment
        Returns a new VideoSegment containing a trimmed version of the original video at the [start, end] segment.
    """

    def __init__(self, video: torch.Tensor, start: int = None, end: int = None, parent_start=0, queues=None):
        """Initializes a VideoSegment object by trimming the video at the given [start, end] times and stores the
        start and end times as attributes. If no times are provided, the video is left unmodified, and the times are
        set to the beginning and end of the video.

        Parameters
        -------
        video : torch.Tensor
            A tensor of the original video.
        start : int
            An int describing the starting frame in this video segment with respect to the original video.
        end : int
            An int describing the ending frame in this video segment with respect to the original video.
        """
        if isinstance(video, VideoSegment):
            # If input is already a VideoSegment, return it as is
            self.trimmed_video = video.trimmed_video
            self.start = video.start
            self.end = video.end
        else:
            if start is None and end is None:
                self.trimmed_video = video
                self.start = 0
                self.end = video.shape[0]  # duration
            else:
                self.trimmed_video = video[start:end]
                if start is None:
                    start = 0
                if end is None:
                    end = video.shape[0]
                self.start = start + parent_start
                self.end = end + parent_start
        print(self.start)
        print(self.end)
        self.num_frames = self.trimmed_video.shape[0]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        if self.trimmed_video.shape[0] == 0:
            raise Exception("VideoSegment has duration=0")

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    def frame_from_index(self, index) -> ImagePatch:
        """Returns the frame at position 'index', as an ImagePatch object."""
        if index < self.num_frames:
            image = self.trimmed_video[index]
        else:
            image = self.trimmed_video[-1]
        return ImagePatch(image)

    def trim(self, start: Union[int, None] = None, end: Union[int, None] = None) -> VideoSegment:
        """Returns a new VideoSegment containing a trimmed version of the original video at the [start, end]
        segment.

        Parameters
        ----------
        start : Union[int, None]
            An int describing the starting frame in this video segment with respect to the original video.
        end : Union[int, None]
            An int describing the ending frame in this video segment with respect to the original video.

        Returns
        -------
        VideoSegment
            a new VideoSegment containing a trimmed version of the original video at the [start, end]
        """
        if start is not None:
            start = max(start, 0)
        if end is not None:
            end = min(end, self.num_frames)

        return VideoSegment(self.trimmed_video, start, end, self.start, queues=self.queues)

    def frame_iterator(self) -> Iterator[ImagePatch]:
        """Returns an iterator over the frames in the video segment."""
        for i in range(self.num_frames):
            yield ImagePatch(self.trimmed_video[i], queues=self.queues)

    def __repr__(self):
        return "VideoSegment({}, {})".format(self.start, self.end)
        
    def recursive_query(self, question: str):
        """Returns the answer to a complicated question asked about the image. 
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        return "recursive_query(" + question + ")"

def select_answer(video, info, question, possible_answers):
    possible_answers = possible_answers.split('[')[1].split(']')[0].split(',')
    print(possible_answers)
    print("entering select answer")
    video_segment = VideoSegment(video)
    first_frame = video_segment.frame_from_index(0)
    first_frame_caption = first_frame.simple_query("Caption: ")
    mid_frame = video_segment.frame_from_index(video_segment.num_frames//2)
    mid_frame_caption = mid_frame.simple_query("Caption: ")
    last_frame = video_segment.frame_from_index(-1)
    last_frame_caption = last_frame.simple_query("Caption: ")
    if not isinstance(info, dict):
        info = {"Original relevent information": str(info)}
    info["Caption of the first frame"]= first_frame_caption
    info["Caption of the middle frame"]= mid_frame_caption
    info["Caption of the last frame"]= last_frame_caption

    prompt = "\
    Given a question asked about an image/video, a dict of information related to the question, and a list of possible answers, \
    choose the most correct answer from the list of possible answers.\n"
    prompt_question = "This is the question asked about this image/video: " + question + "\n\n"
    prompt_info = "These are the info related to the question:\n" 
    prompt_info += str(info)
    prompt_possible_answers = "\n\nChoice: \n"
    letter_list = ['(A) ', "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) ", "(H) ", "(I) ", "(J) ", "(K) ", "(L) ", "(M) ", "(N) "]
    for letter, answer in zip(letter_list, possible_answers):
        print(answer)
        
        prompt_possible_answers += letter + answer + '\n'
    # prompt_possible_answers += str(possible_answers) + "\n"
    prompt_final = "\n\nAnswer (Letter Only):"
    full_prompt = prompt + prompt_question + prompt_info + prompt_possible_answers + prompt_final 
    print(full_prompt)
    response = get_response(full_prompt)
    print(response)
    return response + "###" + str(info)


@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def get_response(input_prompt):
    
    extended_prompt = [input_prompt]
    
    responses = [openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=config.codex.max_tokens,
            top_p = 0.,
            frequency_penalty=0,
            presence_penalty=0,
#                 best_of=config.codex.best_of,
            stop=["\n\n"],
            )
                for prompt in extended_prompt]
    result = [r['choices'][0]['message']['content'].replace("execute_command(image)", "execute_command(image, my_fig, time_wait_between_lines, syntax)") for r in responses]
    
    result = result[0]
        
    # result = codex.check_correctness(orig_code, decomposed_code, result, question, query)
    print("result is:\n"+result)

    return result