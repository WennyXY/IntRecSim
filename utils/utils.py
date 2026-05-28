import os
import random
import yaml
import json
import re
import logging
import numpy as np
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike
from scipy.spatial.distance import cosine
from scipy import stats
from langchain_huggingface import HuggingFaceEmbeddings

def fix_seeds(seed=2026):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)


def load_LLM(config):
    if 'ollama' in config.llm:
        llm = Ollama(
            model = config.llm.replace('ollama:',''),
            temperature = config.temperature,
            request_timeout=200,
            base_url=config.ollama_base_url
        )
    elif 'azure' in config.llm:
        if 'gpt' in config.llm:
            model = config.llm.replace('azure:','')
            print(f'load AzureOpenAI: {model}, temp={config.temperature}')
            llm = AzureOpenAI(
                model = model,
                deployment_name=config.llm.replace('azure:',''),
                temperature = config.temperature,
                api_key=config.azure_api_key,
                azure_endpoint='https://',
                api_version='',
                max_retries=5,
                top_p = config.top_p if hasattr(config, 'top_p') else 1.0,
                seed = config["seed"]
            )
        else:
            print(f'load OpenAILike: {config.llm}')
            llm = OpenAILike(
                max_tokens = 4096,
                model=config.llm.replace('azure:',''),
                temperature = config.temperature,
                api_base='',
                api_key=config.azure_api_key,
                max_retries=5,
                is_chat_model = True,
                is_function_calling_model=False,
                seed = config["seed"]
            )
    else:
        print(f'load OpenAI: {config.llm}')
        llm = OpenAI(
            max_tokens = 1500,
            temperature = config.temperature,
            api_key=config.api_key,
            api_base=config.api_base,
            model = "gpt-4o-mini",
            max_retries = 5,
            seed=config["seed"]
        )
    return llm

def load_client(config):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_version="",
        azure_endpoint="https://",
        api_key=config.azure_api_key,
        azure_deployment=config.llm.replace('azure:',''),
    )
    return client


def get_completion(client, messages, model="", temperature=0.1):
    messages = [to_openai_message_dict(message) for message in messages]
    response = ''
    if 'llama' in model.lower():
        print(f'calling llama: {model}')
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature,
            max_tokens = 4096
        )
    else:
        print(f'calling {model}')
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature = temperature
        )
    return response.choices[0].message.content

def load_prompt(prompt_file):
    with open(prompt_file, 'r') as file:
        prompt_dict = yaml.safe_load(file)
    
    return prompt_dict

def profile_to_prompt(user_profile_dict, dict_keys = []):
    user_info = ''
    for key in user_profile_dict.keys():
        if 'id' in key or 'count' in key or 'score' in key:
            continue
        if len(dict_keys) == 0 or key in dict_keys:
            if 'profile' in key and user_profile_dict[key]:
                tmp = str(user_profile_dict[key]) + '\n'
            else:
                tmp = key + ': ' + str(user_profile_dict[key]) + '\n'
            user_info += tmp
    return user_info

def item_to_prompt(item_dict, dict_keys = []):
    item_info = ''
    for key in item_dict.keys():
        if 'id' in key:
            continue
        if item_dict[key] is None or item_dict[key] == '' or len((item_dict[key])) == 0:
            continue
        if len(dict_keys) == 0 or key in dict_keys:
            tmp = str(key).replace('_',' ') + ': ' + str(item_dict[key]) + '\n'
            item_info += tmp
    return item_info

def recommendation_list_to_prompt(recommendation_list, dict_keys = [], brief = False):
    # format items only keep: title, genres, description, average_rating, number of rating, price
    # print(recommendation_list)
    if brief:
        if 'book_title' in recommendation_list[0].keys():
            dict_keys = ['book_title', 'genre']
        elif 'product_title' in recommendation_list[0].keys():
            dict_keys = ['product_title']
        elif 'title' in recommendation_list[0].keys():
            dict_keys = ['title', 'genres']
        else:
            dict_keys = []
    else:
        if 'book_title' in recommendation_list[0].keys():
            dict_keys = ['book_title', 'genre', 'average_rating', 'rating_number', 'description', 'price']
        elif 'product_title' in recommendation_list[0].keys():
            dict_keys = ['product_title', 'product_brand', 'store', 'average_rating', 'rating_number', 'price']
        elif 'title' in recommendation_list[0].keys():
            dict_keys = ['title', 'genres', 'description', 'average_rating', 'popularity']
        else:
            dict_keys = []
    new_recommendation_list = []
    for idx, item in enumerate(recommendation_list):
        item_info = str(idx + 1) + '. '
        for key in item.keys():
            if 'id' in key:
                continue
            if len(dict_keys) > 0 and key not in dict_keys:
                continue
            if item[key] is None or str(item[key]) == '' or (isinstance(item[key], list) and len(item[key]) == 0):
                continue
            tmp = str(key).replace('_',' ') + ': ' + str(item[key]) + '\n'
            item_info += tmp
        new_recommendation_list.append(item_info)
    
    return new_recommendation_list

def format_response_items(items):
    item_name = ''
    if 'book_title' in items[0].keys():
        item_name = 'book'
    elif 'product_title' in items[0].keys():
        item_name = 'beauty product'
    else:
        item_name = 'movie'
    formatted_items = []
    for item in items:
        if item_name == 'book':
            item_str = "Book title: "+ item['book_title']+" \nGenre: " + item['genre'] + " \nBrief description: " + item['brief_description']+" \nUser Review: "+item["review"]+" \nUser Rating: "+str(item['user_rating'])+"/5; Average Rating: "+str(item['average_rating'])+"/5; Popularity: "+ str(item['rating_number'])
        elif "beauty" in item_name:
            if str(item['price']) == "nan":
                price = ""
            else:
                price = "\nPrice: "+str(item['price']) 
            user_review = "User review: " + item['user_review'].replace("<br />"," ")
            item_str = f"Product: {item['product_title']} {price} \nBrand: {item['product_brand']}; Store: {item['store']} \nUser Review (time: {item['review_time']}): {user_review} \n\nUser Rating: {str(item['user_rating'])}/5; Average Rating: {str(item['average_rating'])}/5; Rating number: {str(item['rating_number'])}"
        else: # movie
            item_str = f"Movie title: {item['title']} \nGenres: {item['genres']} \nMovie description: {item['description']} \nUser Rating: {str(item['user_rating'])}/5; Average Rating: {str(item['average_rating'])}/5 \nPopularity: {str(item['popularity'])}"
        formatted_items.append(item_str)
    return "\n".join(formatted_items)


def format_postive_items(items, item_name='', brief_version = False):
    formatted_items = []
    if len(items) == 0:
        return ""
    if item_name == '':
        if 'book_title' in items[0].keys():
            item_name = 'book'
        elif 'product_title' in items[0].keys():
            item_name = 'beauty product'
        else:
            item_name = 'movie'
    for item in items:
        if 'book' in item_name:
            if brief_version:
                item_str = f"Book title: {item['book_title']}; genre: {item['genre']}; User gave rating: {item['user_rating']}"
            else:
                item_str = f"Book title: {item['book_title']} \nGenre: {item['genre']} \nBrief description: {item['brief_description']} \nUser Review (time: {item['review_time']}): {item['review']} \nUser Rating: {str(item['user_rating'])}/5; Average Rating: {str(item['average_rating'])}/5; Rating number: {str(item['rating_number'])}"
        elif "beauty" in item_name:
            if brief_version:
                item_str = f"Product: {item['product_title']}; Brand: {item['product_brand']}; Store: {item['store']}; User gave rating: {item['user_rating']} "
            else:
                if str(item['price']) == "nan":
                    price = ""
                else:
                    price = "\nPrice: "+str(item['price']) 
                user_review = item['user_review'].replace("<br />"," ")
                item_str = f"Product: {item['product_title']} {price} \nBrand: {item['product_brand']}; Store: {item['store']} \nUser Review (time: {item['review_time']}): {user_review} \nUser Rating: {str(item['user_rating'])}/5; Average Rating: {str(item['average_rating'])}/5; Rating Num: {str(item['rating_number'])}"
        else: # movie
            if brief_version:
                item_str = f"Movie title: {item['title']}; Genres: {item['genres']}; \nUser Rating: {str(item['user_rating'])}/5 "
            else:
                item_str = f"Movie title: {item['title']} \nGenres: {item['genres']} \nMovie description: {item['description']} \nAverage Rating: {str(item['average_rating'])}/5 \nPopularity: {str(item['popularity'])}"
        formatted_items.append(item_str)
    return "\n".join(formatted_items)


def calculate_cosine(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return cosine(embedding1, embedding2)

def get_item_text(item_dict, ignore_keys = ['movieid', 'rating', 'popularity']):
    # all keys contain 'id' will be ignored
    txt = ''
    for key in item_dict.keys():
        if key in ignore_keys or 'id' in key:
            continue
        # if item_dict[key] is a list
        if isinstance(item_dict[key], list):
            txt+= str(key) + ': ' + ', '.join([str(i.strip()) for i in item_dict[key]]) + '. '
        else:
            txt+= str(key) + ': ' + str(item_dict[key]) + '. '
    return txt.replace('_', ' ')

def cal_cosine_distance_pair(profile, positive_items, negative_items):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name="/data/pretrain_dir/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    # profile_str = profile_to_prompt(profile)
    profile_str = profile
    user_profile_embeddings = embedding_model.embed_query(profile_str)
    
    positive_distance = 0
    negative_distance = 0
    
    for item in positive_items:
        positive_item_embeddings = embedding_model.embed_query(item)
        positive_distance += calculate_cosine(user_profile_embeddings, positive_item_embeddings)

    for item in negative_items:
        negative_item_embeddings = embedding_model.embed_query(item)
        negative_distance += calculate_cosine(user_profile_embeddings, negative_item_embeddings)

    return positive_distance/len(positive_items), negative_distance/len(negative_items), len(profile_str)
    

def cal_user_item_distance(ratings_df, users_df, items_df, userid, embed,distance_fn=calculate_cosine):
    np.random.seed(2025)
    users_dict = users_df.set_index('userid').to_dict(orient='index')
    
    interacted_items_df = ratings_df[ratings_df['userid']==userid]
    interacted_items_df = interacted_items_df[['movieid']].merge(items_df, on=['movieid'])
    interacted_items_df.set_index('movieid', inplace=True)
    interacted_items = interacted_items_df.to_dict(orient='index')

    user_profile_str = profile_to_prompt(users_dict[userid])
    user_profile_embeddings = embed.embed_query(user_profile_str)
    positive_dis = 0
    for _, item in interacted_items.items():
        item_embeddings = embed.embed_query(get_item_text(item))
        positive_dis += distance_fn(user_profile_embeddings, item_embeddings)
    ave_positive_dis = positive_dis/len(interacted_items)

    negative_dis = 0
    negative_items_df = items_df[~items_df['movieid'].isin(interacted_items.keys())]
    negative_items_df.set_index('movieid', inplace=True)
    negative_items = negative_items_df.to_dict(orient='index')
    unrated_idx_p = np.array(negative_items_df['popularity'])
    unrated_idx_p = unrated_idx_p/unrated_idx_p.sum()
    random_negative_keys = np.random.choice(list(negative_items.keys()), len(interacted_items), p=unrated_idx_p, replace=0)
    for key in random_negative_keys:
        item_embeddings = embed.embed_query(get_item_text(negative_items[key]))
        negative_dis += distance_fn(user_profile_embeddings, item_embeddings)
    ave_negative_dis = negative_dis/len(random_negative_keys)
    return ave_positive_dis, ave_negative_dis, len(user_profile_str)

def general_parser(response, signal=None):
    if response:
        response = response.split('</think>')[-1]
        if signal:
            response = response.strip().split(signal)[-1]
        return response.strip()
    else:
        return ''

def get_pos_item_text(item):
    return item['description'] + " History rating: " + str(item['average_rating']) + "; Popularity: " + str(int(item['popularity'])) + " people have watched); User's rating: " + str(item['user_rating']) +". "

def json_parser(response):
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    else:
        raise ValueError("No valid JSON block found.")

def list_parser(response):
    return response.split("[attributes]:")[-1].split(",")


def get_overlap_ratio(y_pred, y_list):
    intersection = len(list(set(y_pred).intersection(y_list)))
    return intersection/len(y_list)


# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


def extract_json(text):
    # Match the first JSON object in the text
    start = text.find('{')
    if start == -1:
        return {}

    brace_count = 0
    end = start

    # Scan forward and match braces
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1

        if brace_count == 0:
            end = i + 1
            break

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def json_to_profile(json_profile):
    profile_str = ""
    for key, value in json_profile.items():
        profile_str += f"{key}: {'; '.join(value)}\n"
    return profile_str


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def get_eval_scores(y, pred):
    accuracy = accuracy_score(y, pred)
    precision_macro = precision_score(y, pred, average='macro')
    recall_macro = recall_score(y, pred, average='macro')
    f1_macro = f1_score(y, pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro
    }


def NDCG(y_true, pred_ranked_list, k=5):
    pred_ranked_list = pred_ranked_list[:k]
    relevance = [0] * len(pred_ranked_list)
    if y_true in pred_ranked_list:
        relevance[pred_ranked_list.index(y_true)] = 1

    rel = np.asarray(relevance)
    ideal_dcg = IDCG(rel)
    if ideal_dcg == 0:
        return 0.0

    return DCG(rel) / ideal_dcg

def IDCG(relevance):
    rel = np.asarray(relevance).copy()
    rel.sort()
    return DCG(rel[::-1])

def DCG(relevance):
    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)

    log2i = np.log2(range(2, p + 1))
    return rel[0] + (rel[1:] / log2i).sum()


def hits(y_true, predicted_items, k=3):
    """
    Calculates the hit rate for a list of predicted items.

    Args:
        predicted_items: A list of lists, where each inner list contains the predicted items for a user, ranked from most to least relevant.
        actual_items: A list of lists, where each inner list contains the actual relevant items for a user.
        k: The number of top-k items to consider.

    Returns:
        The hit rate.
    """

    hits = 0
    if y_true in predicted_items[:k]:
        hits += 1
    return hits

def calculate_correlation(ground_truth_ratings, predicted_ratings):
    """
    Calculates Pearson and Spearman correlations between predictions and actuals.
    """
    
    # 1. Pearson Correlation (Linear Relationship)
    # Checks if specific rating values move up/down together linearly.
    pearson_corr, pearson_p_value = stats.pearsonr(predicted_ratings, ground_truth_ratings)
    
    # 2. Spearman's Rank Correlation (Monotonic Relationship)
    # Checks if the Ranking order is preserved (e.g., is Item A ranked higher than Item B?)
    spearman_corr, spearman_p_value = stats.spearmanr(predicted_ratings, ground_truth_ratings, nan_policy='omit')

    return {
        "pearson": pearson_corr,
        "pearson_p_value": pearson_p_value,
        "spearman": spearman_corr,
        "spearman_p_value": spearman_p_value
    }


def calculate_rmse(ground_truth_ratings, predicted_ratings):
    predicted = np.array(predicted_ratings)
    ground_truth = np.array(ground_truth_ratings)
    return np.sqrt(np.mean((predicted - ground_truth) ** 2))


def format_ICL_examples(pos_items, neg_items, task_name):
    if 'discr' in task_name.lower():
        rec_items = pos_items + neg_items
        random.shuffle(rec_items)
        return "You are recommended a list of items: \n" + '\n'.join(recommendation_list_to_prompt(rec_items, brief=True)) + "\n Then the following items are correctly selected: \n" + '\n'.join(recommendation_list_to_prompt(pos_items, brief=True)) + "\n\n"
    elif 'rat' in task_name.lower():
        return "User interacted with the following items: \n" + '\n'.join(recommendation_list_to_prompt(pos_items)) + "\n\n For rating task: \n " + format_postive_items(pos_items, brief_version=True) + "\n\n"
    else:
        rec_items = pos_items[:1] + neg_items
        random.shuffle(rec_items)
        # print(len(rec_items))
        return "User are recommended a list of items: \n" + '\n'.join(recommendation_list_to_prompt(rec_items)) + "\n\n These items should be ranked in the following order: \n 1. " + '\n'.join(recommendation_list_to_prompt(pos_items[:1], brief=True)) + " 2. other item that user didn't choose \n 3. other item that user didn't choose ..."

def quote_process(text):
    return text.replace('"',"'")



# from llama_index
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    AudioBlock,
    DocumentBlock,
    ToolCallBlock,
)

from openai.types.chat import ChatCompletionMessageParam
from typing import Optional

def to_openai_message_dict(
    message: ChatMessage,
    drop_none: bool = False,
    model: Optional[str] = None,
) -> ChatCompletionMessageParam:
    """Convert a ChatMessage to an OpenAI message dict."""
    content = []
    content_txt = ""
    reference_audio_id = None
    for block in message.blocks:
        if message.role == MessageRole.ASSISTANT:
            reference_audio_id = message.additional_kwargs.get(
                "reference_audio_id", None
            )
            # if reference audio id is provided, we don't need to send the audio
            if reference_audio_id:
                continue

        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
            content_txt += block.text
        elif isinstance(block, DocumentBlock):
            if not block.data:
                file_buffer = block.resolve_document()
                b64_string = block._get_b64_string(file_buffer)
                mimetype = block._guess_mimetype()
            else:
                b64_string = block.data.decode("utf-8")
                mimetype = block._guess_mimetype()
            content.append(
                {
                    "type": "file",
                    "filename": block.title,
                    "file_data": f"data:{mimetype};base64,{b64_string}",
                }
            )
        elif isinstance(block, ImageBlock):
            if block.url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(block.url),
                            "detail": block.detail or "auto",
                        },
                    }
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.image_mimetype};base64,{img_str}",
                            "detail": block.detail or "auto",
                        },
                    }
                )
        elif isinstance(block, AudioBlock):
            audio_bytes = block.resolve_audio(as_base64=True).read()
            audio_str = audio_bytes.decode("utf-8")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_str,
                        "format": block.format,
                    },
                }
            )
        elif isinstance(block, ToolCallBlock):
            try:
                function_dict = {
                    "type": "function",
                    "function": {
                        "name": block.tool_name,
                        "arguments": block.tool_kwargs,
                    },
                    "id": block.tool_call_id,
                }

                if len(content) == 0 or content[-1]["type"] != "text":
                    content.append(
                        {"type": "text", "text": "", "tool_calls": [function_dict]}
                    )
                elif content[-1]["type"] == "text" and "tool_calls" in content[-1]:
                    content[-1]["tool_calls"].append(function_dict)
                elif content[-1]["type"] == "text" and "tool_calls" not in content[-1]:
                    content[-1]["tool_calls"] = [function_dict]
            except Exception:
                print(
                    f"It was not possible to convert ToolCallBlock with call id {block.tool_call_id or '`no call id`'} to a valid message, skipping..."
                )
                continue
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    # NOTE: Sending a null value (None) for Tool Message to OpenAI will cause error
    # It's only Allowed to send None if it's an Assistant Message and either a function call or tool calls were performed
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    already_has_tool_calls = any(
        isinstance(block, ToolCallBlock) for block in message.blocks
    )
    content_txt = (
        None
        if content_txt == ""
        and message.role == MessageRole.ASSISTANT
        and (
            "function_call" in message.additional_kwargs
            or "tool_calls" in message.additional_kwargs
            or already_has_tool_calls
        )
        else content_txt
    )

    # If reference audio id is provided, we don't need to send the audio
    # NOTE: this is only a thing for assistant messages
    if reference_audio_id:
        message_dict = {
            "role": message.role.value,
            "audio": {"id": reference_audio_id},
        }
    else:
        # NOTE: Despite what the openai docs say, if the role is ASSISTANT, SYSTEM
        # or TOOL, 'content' cannot be a list and must be string instead.
        # Furthermore, if all blocks are text blocks, we can use the content_txt
        # as the content. This will avoid breaking openai-like APIs.
        message_dict = {
            "role": message.role.value,
            "content": (
                content_txt
                if message.role.value in ("assistant", "tool", "system")
                or all(isinstance(block, TextBlock) for block in message.blocks)
                else content
            ),
        }
        if already_has_tool_calls:
            existing_tool_calls = []
            for c in content:
                existing_tool_calls.extend(c.get("tool_calls", []))

            if existing_tool_calls:
                message_dict["tool_calls"] = existing_tool_calls

    if (
        "tool_calls" in message.additional_kwargs
        or "function_call" in message.additional_kwargs
    ) and not already_has_tool_calls:
        message_dict.update(message.additional_kwargs)

    if "tool_call_id" in message.additional_kwargs:
        message_dict["tool_call_id"] = message.additional_kwargs["tool_call_id"]

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    return message_dict  # type: ignore