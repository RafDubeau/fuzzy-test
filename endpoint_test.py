from typing import List
import os
import json

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chat_models import ChatOpenAI
from test_utils import *

from PythonFuzzyTypes import validateEnhancedExperienceVideoData


def generate_video_data(amount=10, verbose=False) -> list[dict]:
    # Set up langchain calls to generate sample video data
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    class_text = """
class EnhancedExperienceVideoData(TypedDict, total=False):
    productDescription: Required[str]
    productHighlights: Required[List[str]]
    productItineraryItems: Required[List[str]]
    productLanguage: Required[str]
    productLocation: Required[str]
    productTitle: Required[str]
    sourceVideoDescription: str
    videoIntelligenceDescription: Required[str]
    videoIntelligenceObjects: Required[Dict[str, Union[int, float]]]
    videoTextOnScreen: Required[List[str]]
    videoTranscript: str
"""

    video_data_prompt = PromptTemplate.from_template(
        template="""
    You are a system designed to produce video data for a travel experience platform.
                                   
    Here is the python TypedDict for the EnhancedExperienceVideoData data type:
    {data_format}

    Generate a COMPLETE list of {amount} EnhancedExperienceVideoData objects, of sample experience videos for a variety of different types of vacations that matches this data type.

    Output ONLY the list of python dict objects. Do not include variable assignment or print statements.                      
    """
    )

    raw_data_chain = LLMChain(
        llm=llm, prompt=video_data_prompt, output_key="raw_data", verbose=verbose
    )

    def parse_json(inputs: dict) -> dict:
        return {"data_dict": eval(inputs["raw_data"])}

    parse_json_chain = TransformChain(
        input_variables=["raw_data"],
        output_variables=["data_dict"],
        transform=parse_json,
        verbose=verbose,
    )

    generation_chain = SequentialChain(
        input_variables=["data_format", "amount"],
        output_variables=["data_dict"],
        chains=[raw_data_chain, parse_json_chain],
        verbose=verbose,
    )

    result = generation_chain.run(data_format=class_text, amount=amount)

    return result


def get_video_data(video_ids: list[str]) -> dict[str, dict]:
    video_data = {}

    cache_folder = "cached_video_data"

    uncached_video_ids = []
    for video_id in video_ids:
        cached_data_path = os.path.join(cache_folder, f"{video_id}.json")
        if os.path.exists(cached_data_path):
            with open(cached_data_path, "r") as f:
                data = json.load(f)
                video_data[video_id] = data
        else:
            uncached_video_ids.append(video_id)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    while len(uncached_video_ids) > 0:
        batch_cutoff = min(5, len(uncached_video_ids))
        gen_video_ids, uncached_video_ids = (
            uncached_video_ids[:batch_cutoff],
            uncached_video_ids[batch_cutoff:],
        )

        gen_video_data = generate_video_data(amount=len(gen_video_ids), verbose=False)

        for video_id, data in zip(gen_video_ids, gen_video_data):
            data = validateEnhancedExperienceVideoData(data)
            video_data[video_id] = data

            cached_data_path = os.path.join(cache_folder, f"{video_id}.json")

            with open(cached_data_path, "w") as f:
                json.dump(data, f, indent=4)

    return video_data


def initialize_videos(video_ids: List[str]):
    video_data = get_video_data(video_ids)

    for id, data in video_data.items():
        post(
            "add_experience_video",
            experience_video_id=id,
            data=data,
        )


def initialize_users(user_ids: List[str]):
    for user_id in user_ids:
        post("add_user_to_recommendation_engine", user_id=user_id)


def get_cf_recommendation_distribution(
    user_groups: List[List[str]],
    video_groups: List[List[str]],
    num_recommendations: int = 10,
):
    recommendation_results = []
    avg_scores = []
    for user_group in user_groups:
        recommendation_results.append([0] * len(video_groups))
        avg_scores.append([0] * len(video_groups))

        for user_id in user_group:
            # Get recommendations for user
            response = get(
                "get_cf_recommendations",
                user_id=user_id,
                num_recommendations=num_recommendations,
            )
            video_ids, scores = response["video_ids"], response["scores"]

            # Add to results
            for j, video_id in enumerate(video_ids):
                for i, video_group in enumerate(video_groups):
                    if video_id in video_group:
                        recommendation_results[-1][i] += 1
                        avg_scores[-1][i] += scores[j]
                        break

        # Normalize
        total = sum(recommendation_results[-1])
        recommendation_results[-1] = [
            count / total for count in recommendation_results[-1]
        ]
        avg_scores[-1] = [
            score / recommendation_results[-1][i]
            for i, score in enumerate(avg_scores[-1])
        ]

    for i, user_group in enumerate(recommendation_results):
        print(f"User group {i + 1}: {user_group}  {avg_scores[i]}")

    return recommendation_results


def print_distribution():
    scores = get("compute_cf_scores", verbose=True)

    watch_histories = get(
        "get_interaction_histories", verbose=True, user_ids=all_user_ids
    )

    ## Get average scores for each user and video group
    avg_scores = []
    for user_group in user_groups:
        avg_scores.append([0] * len(video_groups))
        total_counts = [0] * len(video_groups)

        for user_id in user_group:
            for i, video_group in enumerate(video_groups):
                for video_id in video_group:
                    # Filter out predicted scores of watched videos
                    if video_id not in watch_histories[user_id]:
                        total_counts[i] += 1
                        avg_scores[-1][i] += scores[user_id][video_id]

        # Normalize
        avg_scores[-1] = [
            avg_scores[-1][i] / total_counts[i] for i in range(len(video_groups))
        ]

    for i, user_group in enumerate(avg_scores):
        print(f"\nUser group {i+1}:")
        for j, score in enumerate(user_group):
            print(f"\tVideo group {j + 1}: {score}")
    print()


print(f"n_users: {len(all_user_ids)}\nn_videos: {len(all_video_ids)}\n")


def cf_main():
    ## Clear watch data
    post("clear_interaction_data")

    ## Add random watch events
    post(
        "simulate_interaction_events",
        user_ids=all_user_ids,
        video_ids=all_video_ids,
        num_events=n_users * n_videos // 3,
    )

    post("full_train_cf")

    print_distribution()

    ## Split users into two groups
    post(
        "simulate_interaction_events",
        user_ids=user_groups[0],
        video_ids=video_groups[0],
        num_events=2 * len(user_groups[0]) * len(video_groups[0]) // 3,
        lower_bound=0.9,
    )

    post(
        "simulate_interaction_events",
        user_ids=user_groups[1],
        video_ids=video_groups[1],
        num_events=2 * len(user_groups[1]) * len(video_groups[1]) // 3,
        lower_bound=0.9,
    )

    # ## Train collaborative filtering model
    post("full_train_cf")

    print_distribution()


def animate_main():
    ## Clear watch data
    post("clear_interaction_data")
    post("clear_cf_bucket")

    ## Add random watch events
    post(
        "simulate_interaction_events",
        user_ids=all_user_ids,
        video_ids=all_video_ids,
        num_events=n_users * n_videos // 3,
    )

    post("full_train_cf", save_path="cf_model_0.pkl")

    random_plot = 10 * [
        (user_groups[0], video_groups[0], 50, 0.85, 1),
        (user_groups[1], video_groups[1], 50, 0.85, 1),
    ]

    plot = random_plot
    n = len(plot)

    for i, scene in enumerate(plot):
        disp_str = f"-------------------- Scene {i+1:<2} --------------------"
        print(disp_str)
        post(
            "simulate_interaction_events",
            user_ids=scene[0],
            video_ids=scene[1],
            num_events=scene[2],
            lower_bound=scene[3],
            upper_bound=scene[4],
        )

        post(
            "partial_train_users",
            load_path=f"cf_model_{2 * i}.pkl",
            save_path=f"cf_model_{2 * i + 1}.pkl",
        )

        post(
            "partial_train_videos",
            load_path=f"cf_model_{2 * i + 1}.pkl",
            save_path=f"cf_model_{2 * i + 2}.pkl",
        )

        print("-" * len(disp_str))

    post(
        "generate_factor_animation",
        load_paths=[f"cf_model_{i}.pkl" for i in range(n * 2 + 1)],
        save_path="factor_animation.gif",
    )


def hybrid_main():
    christian_id = "B8MajLczOkPOnQvYyKCLGhseexn1"
    rafael_id = "LzGW0uiWrVZAdvvVmAVnE6rqZt03"
    kyle_id = "kPtLzCNeMLNvSLutIzehnQIl7mf1"
    ## Clear data
    # post("clear_user_data")
    # post("clear_experience_video_data")
    # post("clear_weaviate")
    # post("clear_cf_files")
    # post("clear_interaction_data")

    # id_list = get("get_weaviate_id_list", verbose=False)
    # print(f"Num videos in weaviate: {len(id_list)}")
    # print(id_list)

    ## Create users
    # initialize_users(all_user_ids)  # Christian's code replaces this

    ## Create videos
    # initialize_videos(all_video_ids)  # August's code replaces this

    # print("Waiting for Weaviate to generate embeddings...")

    # with tqdm(total=len(all_video_ids)) as pbar:
    #     n_ids = len(get("get_weaviate_id_list", verbose=False))
    #     pbar.update(n_ids - pbar.n)
    #     while n_ids < len(all_video_ids):
    #         time.sleep(1)
    #         n_ids = len(get("get_weaviate_id_list", verbose=False))
    #         pbar.update(n_ids - pbar.n)
    # print("Weaviate initialization complete! Continuing with script...")

    ## Add random interaction events
    # post(
    #     "simulate_interaction_events",
    #     user_ids=[christian_id, rafael_id],
    #     video_ids=id_list,
    #     num_events=1,
    # )

    ## Initialize the affiliation coefficients
    # post(
    #     "set_affiliation_coefficients",
    #     affiliation_coefficients={
    #         "bookmarked": 0.85,
    #         "unbookmarked": 0.85,
    #         "watched": 0.3,
    #         "shared": 0.75,
    #         "liked": 1,
    #         "unliked": 1,
    #         "viewed-product": 0.85,
    #         "added-to-timeline": 1,
    #         "shared-timeline": 0.75,
    #         "scrolled-details": 0.5,
    #         "scrolled-images": 0.5,
    #         "added-comment": 0.4,
    #         "sent-message": 0.4,
    #         "bought-product": 2,
    #     },
    # )

    ## Train collaborative filtering model
    # post("full_train_cf")

    # Get recommendations
    # recommendations = get(
    #     "get_recommendations", user_id=kyle_id, num_recommendations=20
    # )

    # for rec in recommendations:
    #     if rec not in id_list:
    #         print(f"Recommendation {rec} not found in weaviate")


def bookmark_test():
    post(
        "add_interaction_event",
        interactionType="unbookmarked",
        userId="B8MajLczOkPOnQvYyKCLGhseexn1",
        referenceType="ExperienceVideo",
        referenceId="0c80ac06-a946-4f18-be03-1fe7b5e2d999",
        routerPath="test",
        value=-1,
    )


def backend_recommendation_test():
    kyle_id = "kPtLzCNeMLNvSLutIzehnQIl7mf1"
    truman_id = "KOyDDN84tvTZzwzqFRM0pdz2BlD3"
    get("get_weaviate_id_list", verbose=True)
    post("sync_weaviate", verbose=True)
    get("get_weaviate_id_list", verbose=True)

    # post("clear_cf_files")

    ## Initialize the affiliation coefficients
    # post(
    #     "set_affiliation_coefficients",
    #     affiliation_coefficients={
    #         "bookmarked": 0.85,
    #         "unbookmarked": 0.85,
    #         "watched": 0.3,
    #         "shared": 0.75,
    #         "liked": 1,
    #         "unliked": 1,
    #         "viewed-product": 0.85,
    #         "added-to-timeline": 1,
    #         "shared-timeline": 0.75,
    #         "scrolled-details": 0.5,
    #         "scrolled-images": 0.5,
    #         "added-comment": 0.4,
    #         "sent-message": 0.4,
    #         "bought-product": 2,
    #         "opened-app": 0,
    #         "closed-app": 0,
    #     },
    # )

    ## Train collaborative filtering model
    # post("full_train_cf")

    ## Request recommendations

    # post(
    #     "add_interaction_event",
    #     interactionType="opened-app",
    #     userId=kyle_id,
    #     referenceType="None",
    #     referenceId="",
    #     routerPath="test",
    #     value=1,
    # )

    # for recommended_id in [
    #     "7a41a81c-fdda-4fcb-aac7-65bd93f99992",
    #     "539809c0-7b8e-46e2-8152-406cb94594cf",
    #     "b9b2431c-f83b-4e2b-ae4a-8ed3de76b4a8",
    #     "b1eee97d-2228-4e53-887b-e6124f397b13",
    #     "2b3648ee-3b77-4ac2-ba94-3202c8b946a8",
    #     "34334800-915c-4200-bfa2-6ce6513cfdd8",
    #     "c91ea4ec-635e-424d-acde-de1ad13448a4",
    # ]:
    #     post(
    #         "add_interaction_event",
    #         interactionType="watched",
    #         userId=kyle_id,
    #         referenceType="RecommendedExperienceVideo",
    #         referenceId=recommended_id,
    #         routerPath="test",
    #         value=1,
    #     )

    # post("simulate_scrolling", user_id=kyle_id, num_watched=8)

    # post(
    #     "add_interaction_event",
    #     interactionType="closed-app",
    #     userId=kyle_id,
    #     referenceType="None",
    #     referenceId="",
    #     routerPath="test",
    #     value=1,
    # )


def generate_trending():
    post("request_new_trending_videos")


def suggestion_test(
    user_id: str = "kPtLzCNeMLNvSLutIzehnQIl7mf1",
    wishtrip_id: str = "7c691c25-659c-4c5f-b417-ab7feec9b4c8",
):
    # post("clear_wishtrip_items", wishtrip_id=wishtrip_id)

    post("add_sample_wishtrip", user_id=user_id, wishtrip_id=wishtrip_id)

    # post("clear_wishtrip_suggestions", wishtrip_id=wishtrip_id)

    # post(
    #     "add_wishtrip_suggestions",
    #     wishtrip_id=wishtrip_id,
    # )


if __name__ == "__main__":
    suggestion_test()


"""

firebase functions:delete simulate_interaction_events remove_experience_video add_user_to_recommendation_engine clear_user_data sync_weaviate partial_train_users get_cf_recommendations clear_cf_bucket get_weaviate_id_list plot_factors remove_video_from_recommendation get_recommendations full_train_cf add_interaction_event add_experience_video remove_video_from_recommendation_engine get_user_interaction_histories get_cb_recommendations clear_interaction_data compute_cf_scores query_videos_by_text remove_user_from_recommendation_engine clear_firestore generate_factor_animation partial_train_videos set_affiliation_coefficients add_video_to_recommendation_engine




git+https://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@github.com/bright-blue-im/FuzzyTypes.git


fuzzy-types @ git+https://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@github.com/bright-blue-im/FuzzyTypes.git@62334349271d6a8b3e0dee4f0baeb450da0c875f

"""
