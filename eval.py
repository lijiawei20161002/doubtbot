import dataclasses
from typing import List

from model_wrappers import (
    Llama2Wrapper,
    Llama3Wrapper,
    ModelWrapper,
)
from tqdm import tqdm

from data import QuestionAnswerPair, load_data, save_to_json


def run_critic_eval(
    critic: ModelWrapper,
    judge: ModelWrapper,
    dataset: List[QuestionAnswerPair],
    output_path: str,
):
    results = []
    for item in tqdm(dataset):
        critique = critic.critique(item)
        judge_prob_pre_critique = judge.get_judge_confidence(item)
        judge_prob_post_critique = judge.get_judge_confidence(item, critique=critique)
        results.append(
            {
                "question_answer_pair": dataclasses.asdict(item),
                "critic": critic.model_id,
                "judge": judge.model_id,
                "critique": critique,
                "judge_confidence_pre_critique": judge_prob_pre_critique,
                "judge_confidence_post_critique": judge_prob_post_critique,
            }
        )
        save_to_json(results, output_path)
