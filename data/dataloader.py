import pandas as pd
import numpy as np
# import utils as utils
import random
from collections import defaultdict


class ML_1M:
    def __init__(self, config):
        self.config = config
        self.ratings_df = pd.read_csv(config['ml_1m_rating'])
        self.ratings_df.rename(columns={"rating": "user_rating"}, inplace=True)
        self.items_df = self.load_items_df(config['ml_1m_item'])
        self.users_df = pd.read_csv(config['ml_1m_user'])

    def load_items_df(self, item_path):
        items_df = pd.read_csv(item_path)
        items_df.rename(columns={"rating":"average_rating"}, inplace=True)
        # items_df['title'] = 'item '+ items_df['movieid'].astype(str)
        # items_df = items_df.drop(columns=['description'])
        return items_df
    
    
    def get_interacted_items(self, userid, mergeitem=True):
        interacted_items_df = self.ratings_df[self.ratings_df['userid']==userid]
        if mergeitem:
            item_key = 'itemid' if 'itemid' in self.items_df.columns else 'movieid'
            interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=[item_key])
        return interacted_items_df.sort_values(by='timestamp')

    def get_items_for_profile(self, userid, pos_num=10, neg_num=10, test_num=0, test_sampling='sequence'):
        """Return positive and negative candidate items for an ML-1M user.

        Args:
            userid: ML-1M user id.
            pos_num: Take the last pos_num items from the first 80% of interactions.
            neg_num: Number of randomly sampled non-interacted items.
            test_num: Items from the held-out 20%. -1 returns all; 0 omits test items.
            test_sampling: 'sequence' (chronological) or 'random'.

        Returns:
            (positive_items, negative_items) or (positive_items, negative_items, test_pos_items).
        """
        userid = int(userid)
        interacted_items_df = self.ratings_df[self.ratings_df['userid']==userid]
        interacted_items_df = interacted_items_df[['movieid','user_rating','timestamp']].merge(self.items_df, on=['movieid'])
        sorted_interacted_items_df = interacted_items_df.sort_values(by='timestamp')
        sorted_interacted_items_df.set_index('movieid', inplace=True)
        sorted_interacted_items = sorted_interacted_items_df.to_dict(orient='index')
        negative_items_df = self.items_df[~self.items_df['movieid'].isin(sorted_interacted_items.keys())]
        negative_items_df.set_index('movieid', inplace=True)
        negative_items = negative_items_df.to_dict(orient='index')
        
        # sort based on popularity
        # unrated_idx_p_1 = np.array(negative_items_df['popularity'])
        # unrated_idx_p_1 = unrated_idx_p_1/unrated_idx_p_1.sum()
        # sort based on average_rating
        # unrated_idx_p_2 = np.array(negative_items_df['average_rating'])
        # unrated_idx_p_2 = unrated_idx_p_2/unrated_idx_p_2.sum()
        
        
        # random_negative_keys = np.random.choice(list(negative_items.keys()), neg_num, p=(unrated_idx_p_2 + unrated_idx_p_1)/2, replace=False)
        
        # Fully random negative items
        random_negative_keys = np.random.choice(list(negative_items.keys()), neg_num, replace=False)
        
        negative_items = [negative_items[key] for key in random_negative_keys]
       
        generation_count = int(len(sorted_interacted_items)*0.8)
        positive_items = list(sorted_interacted_items.values())[:generation_count]
        if pos_num > 0:
            positive_items = positive_items[-pos_num:]
        if test_num == -1:
            test_pos_items = list(sorted_interacted_items.values())[generation_count:]
            return positive_items, negative_items, test_pos_items
        if test_num > 0:
            if 'seq' in test_sampling:
                test_pos_items = list(sorted_interacted_items.values())[generation_count:generation_count+test_num]
            else:
                test_pos_items = random.sample(list(sorted_interacted_items.values())[generation_count:], min(test_num, len(sorted_interacted_items)-generation_count))
            return positive_items, negative_items, test_pos_items
        else:
            return positive_items, negative_items
    
    def get_demographic_information(self, userid):
        return ""
    
    def get_data_shape(self):
        return {
            'Ratings': self.ratings_df.shape, 
            'Items':self.items_df.shape, 
            'Users': self.users_df.shape
        }
    
    def get_data_features(self):
        return {
            'Ratings': self.ratings_df.columns, 
            'Items':self.items_df.columns, 
            'Users': self.users_df.columns
        }
        
    def get_user_profile(self, userid):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        return users_dict[userid]
    
    def get_users_df(self):
        return self.users_df
    
    def get_items_df(self):
        return self.items_df
    
    def get_interactions_df(self):
        return self.ratings_df
    

class Amazon_Book:
    def __init__(self, config):        
        self.config = config        
        self.items_df = self.load_items_df(config['amazon_book_item'])
        self.interactions_df = self.load_interactions_df(config['amazon_book_review'])
        self.users_df = pd.read_csv(config['amazon_book_user'])
        
        self.users_df['userid'] = self.users_df['userid'].astype(str)
    
    def load_interactions_df(self, interactions_path):
        interactions_df = pd.read_parquet(interactions_path)
        interactions_df = interactions_df.rename(columns={'title':'review_title', 'images':'review_images','text':'review_content', 'rating':'user_rating', 'timestamp':'review_time'})
        interactions_df['review'] = interactions_df['review_title'] + ". " + interactions_df['review_content']
        review_features = ['userid', 'parent_asin', 'review', 'user_rating', 'review_time']
        interactions_df['parent_asin'] = interactions_df['parent_asin'].astype(str)
        interactions_df['userid'] = interactions_df['userid'].astype(str)
        return interactions_df[review_features]
        
    def load_items_df(self, items_path):
        items_df = pd.read_parquet(items_path)
        items_df = items_df.rename(columns={'title':'book_title', 'images':'book_images'})
        
        # extract genres
        def get_genres(x):
            genres = ','.join(x[1:]) if len(x)>0 else x
            return genres
        items_df['genre'] = items_df['categories'].apply(get_genres)
        
        items_features = ['parent_asin', 'book_title', 'genre', 'average_rating', 'rating_number', 'brief_description', 'price']
        # items_features = ['parent_asin', 'book_title', 'genre', 'average_rating', 'rating_number', 'price']
        items_df= items_df[items_features]
        items_df['parent_asin'] = items_df['parent_asin'].astype(str)
        return items_df

    def get_interacted_items(self, userid, mergeitem=True):
        userid = str(userid)
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        if mergeitem:
            interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=['parent_asin'])
        return interacted_items_df.sort_values(by='review_time')
    
    def get_items_for_profile(self, userid, pos_num=13, neg_num=10, test_num=0, test_sampling='sequence'):
        """Return positive and negative candidate items for an Amazon Book user.

        Args:
            userid: Book user id.
            pos_num: Take the last pos_num items from the first 80% of interactions.
            neg_num: Number of popularity-sampled non-interacted items.
            test_num: Items from the held-out 20%. -1 returns all; 0 omits test items.
            test_sampling: 'sequence' (chronological) or 'random'.

        Returns:
            (positive_items, negative_items) or (positive_items, negative_items, test_pos_items).
        """
        userid = str(userid)
        unified_features = ['book_title', 'genre', 'average_rating', 'rating_number', 'price', 'parent_asin', 'brief_description']
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        interacted_items_df = interacted_items_df.merge(self.items_df[unified_features], how='inner', on=['parent_asin'])
        interacted_items_df = interacted_items_df[['book_title', 'genre', 'average_rating', 'rating_number', 'price', 'review', 'user_rating', 'review_time', 'parent_asin', 'brief_description']]
        sorted_interacted_items_df = interacted_items_df.sort_values(by='review_time')
        sorted_interacted_items_df.set_index('parent_asin', inplace=True)
        sorted_interacted_items = sorted_interacted_items_df.to_dict(orient='index')
        # print(len(sorted_interacted_items))
        negative_items_df = self.items_df[~self.items_df['parent_asin'].isin(sorted_interacted_items.keys())]
        negative_items_df = negative_items_df[unified_features]
        negative_items_df.set_index('parent_asin', inplace=True)
        negative_items = negative_items_df.to_dict(orient='index')
        
        # ====== Popularity Sampling
        unrated_idx_p = np.array(negative_items_df['rating_number'])
        unrated_idx_p = unrated_idx_p/unrated_idx_p.sum()
        # ====== Average_rating Sampling
        # unrated_idx_p = np.array(negative_items_df['average_rating'])
        # unrated_idx_p = unrated_idx_p/unrated_idx_p.sum()
        random_negative_keys = np.random.choice(list(negative_items.keys()), neg_num, p=unrated_idx_p, replace=False)
        # random_negative_keys = np.random.choice(list(negative_items.keys()), neg_num, replace=False)
        negative_items = [negative_items[key] for key in random_negative_keys]
        
        # Top Popularity
        # negative_items = sorted(negative_items.values(), key=lambda item: item['rating_number'], reverse=True)
        # negative_items = negative_items[:neg_num]
        # print('top popular: ', negative_items)
        
        # first 80% for generation
        generation_count = int(len(sorted_interacted_items)*0.8)
        positive_items = list(sorted_interacted_items.values())[:generation_count]
        # eval_pos_items = list(sorted_interacted_items.values())[generation_count:generation_count+eval_num]
        if pos_num > 0:
            positive_items = list(sorted_interacted_items.values())[:generation_count][-pos_num:]
        if test_num == -1:
            test_pos_items = list(sorted_interacted_items.values())[generation_count:]
            return positive_items, negative_items, test_pos_items
        if test_num > 0:
            if 'seq' in test_sampling:
                test_pos_items = list(sorted_interacted_items.values())[generation_count:generation_count+test_num]
            else:
                test_pos_items = random.sample(list(sorted_interacted_items.values())[generation_count:], min(test_num, len(sorted_interacted_items)-generation_count))    
            return positive_items, negative_items, test_pos_items
        return positive_items, negative_items
        
    # def review_processing(self, reviews_df):
    #     # reduce length of review content
    #     review_text = reviews_df['review_title'] + ". " + reviews_df['review_content']
    #     return review_text
    
    def get_demographic_information(self, userid, activity=False):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        if activity and 'activity' in users_dict[userid].keys():
            return users_dict[userid]['activity']
        return ''
    
    def get_user_profile(self, userid):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        return users_dict[userid]
    
    def get_data_shape(self):
        return {
            'Interactions': self.interactions_df.shape, 
            'Items':self.items_df.shape
        }
    
    def get_data_features(self):
        return {
            'Interactions': self.interactions_df.columns, 
            'Items':self.items_df.columns
        }
    
    def get_users_df(self):
        return self.users_df
    
    def get_items_df(self):
        return self.items_df
    
    def get_interactions_df(self):
        return self.interactions_df


class Amazon_Beauty:
    def __init__(self, config):        
        self.config = config        
        self.items_df = self.load_items_df(config['amazon_beauty_item'])
        self.interactions_df = self.load_interactions_df(config['amazon_beauty_review'])
        self.users_df = pd.read_csv(config['amazon_beauty_user'])
    
    def load_interactions_df(self, interactions_path):
        interactions_df = pd.read_csv(interactions_path)
        interactions_df = interactions_df.rename(columns={'title':'review_title','text':'review_content', 'rating':'user_rating', 'timestamp':'review_time'})
        interactions_df['user_review'] = interactions_df['review_title'] + ". " + interactions_df['review_content']
        interactions_df['interaction_id'] = interactions_df.index
        review_features = ['userid', 'parent_asin', 'user_review', 'user_rating', 'review_time','interaction_id']
        return interactions_df[review_features]
        
    def load_items_df(self, items_path):
        import ast
        if 'csv' in items_path:
            items_df = pd.read_csv(items_path)
        else:
            items_df = pd.read_parquet(items_path)
        items_df = items_df.rename(columns={'title':'product_title', "features":"product_features", 'description':'product_description', 'details':'product_details'})
        # extract product brand
        items_df['product_details'] = items_df['product_details'].astype(str).apply(ast.literal_eval)
        items_df['product_brand'] = items_df['product_details'].apply(lambda x: x['Brand'] if 'Brand' in x else 'Unknown')
        
        items_features = ['parent_asin', 'product_title', 'average_rating', 'rating_number','price', 'store','product_brand']
        # items_df['brand'] = items_df['product_details']['brand']
        # , "product_features", "product_details", "product_description"
        items_df= items_df[items_features]
        return items_df
        
    def get_interacted_items(self, userid, mergeitem):
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        if mergeitem:
            interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=['parent_asin'])
        return interacted_items_df.sort_values(by='review_time')
        
    def get_items_for_profile(self, userid, pos_num=10, neg_num=10, test_num=0, test_sampling='sequence'):
        """Return positive and negative candidate items for an Amazon Beauty user.

        Args:
            userid: Beauty user id.
            pos_num: Take the last pos_num items from the first 80% of interactions.
            neg_num: Number of randomly sampled non-interacted items.
            test_num: Items from the held-out 20%. -1 returns all; 0 omits test items.
            test_sampling: 'sequence' (chronological) or 'random'.

        Returns:
            (positive_items, negative_items) or (positive_items, negative_items, test_pos_items).
        """
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        interacted_items_df = interacted_items_df.drop(columns=['userid'])
        interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=['parent_asin'])
        
        sorted_interacted_items_df = interacted_items_df.sort_values(by='review_time')
        sorted_interacted_items_df = sorted_interacted_items_df.drop(columns=['interaction_id'], errors='ignore')
        # Drop duplicate item interactions (same parent_asin rated twice); keep the latest.
        sorted_interacted_items_df = sorted_interacted_items_df.drop_duplicates(subset=['parent_asin'], keep='last')
        sorted_interacted_items_df.set_index('parent_asin', inplace=True)
        sorted_interacted_items = sorted_interacted_items_df.to_dict(orient='index')
        negative_items_df = self.items_df[~self.items_df['parent_asin'].isin(sorted_interacted_items_df.index.to_list())]
        negative_items_df.set_index('parent_asin', inplace=True)
        negative_items = negative_items_df.to_dict(orient='index')
        
        # popularity sampling
        # unrated_idx_p = np.array(negative_items_df['rating_number'])
        # unrated_idx_p = unrated_idx_p/unrated_idx_p.sum()
        # random_negative_keys = np.random.choice(list(negative_items.keys()), len(sorted_interacted_items), p=unrated_idx_p, replace=0)
        
        # fully random
        random_negative_keys = np.random.choice(list(negative_items.keys()), neg_num, replace=False)
        negative_items = [negative_items[key] for key in random_negative_keys]
        
        # first 80% for generation, next 20% to provide signal for optimisation during profile generation and for testing.
        generation_count = int(len(sorted_interacted_items)*0.8)
        positive_items = list(sorted_interacted_items.values())[:generation_count]
        if pos_num > 0:
            positive_items = list(sorted_interacted_items.values())[:generation_count][-pos_num:]
        if test_num == -1:
            test_pos_items = list(sorted_interacted_items.values())[generation_count:]
            return positive_items, negative_items, test_pos_items
        if test_num > 0:
            if 'seq' in test_sampling:
                test_pos_items = list(sorted_interacted_items.values())[generation_count:generation_count+test_num]
            else:
                test_pos_items = random.sample(list(sorted_interacted_items.values())[generation_count:], min(test_num, len(sorted_interacted_items)-generation_count))    
            return positive_items, negative_items, test_pos_items
        else:
            return positive_items, negative_items

    def get_demographic_information(self, userid, activity=False):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        if activity and 'activity' in users_dict[userid].keys():
            return users_dict[userid]['activity']
        return ''
    
    def get_user_profile(self, userid):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        return users_dict[userid]
    
    def get_data_shape(self):
        return {
            'Interactions': self.interactions_df.shape, 
            'Items':self.items_df.shape
        }
    
    def get_data_features(self):
        return {
            'Interactions': self.interactions_df.columns, 
            'Items':self.items_df.columns
        }
    
    def get_users_df(self):
        return self.users_df
    
    def get_items_df(self):
        return self.items_df
    
    def get_interactions_df(self):
        return self.interactions_df





class Steam:
    def __init__(self, config):
        self.config = config
        self.interactions_df = self.load_interactions_df(config['steam_review'])
        self.items_df = self.load_items_df(config['steam_item'])
        self.users_df = pd.read_csv(config['steam_user'])
    
    def load_interactions_df(self, interactions_path):
        interactions_df = pd.read_csv(interactions_path)
        interactions_df = interactions_df.rename(columns={'text':'review', 'date':'review_date'})
        interactions_df = interactions_df[['interactionid', 'userid', 'itemid', 'review', 'review_date']]
        return interactions_df
    
    def load_items_df(self, items_path):
        items_df = pd.read_csv(items_path).dropna(subset=['itemid']).drop_duplicates(subset="itemid", keep='last')
        items_df = items_df.rename(columns={'title':'game_name', 'genres':'game_genres','early_access':'early_access_support'})
        items_df = items_df[['itemid', 'game_name', 'game_genres', 'tags', 'release_date', 'publisher', 'specs']]
        
        return items_df
    
    def get_interacted_items(self, userid, mergeitem=True):
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        if mergeitem:
            interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=['itemid'])
        # 'date' is renamed to 'review_date' in load_interactions_df
        return interacted_items_df.sort_values(by='review_date')
    
    def get_users_df(self):
        return self.users_df
    
    def get_items_for_profile(self, userid, pos_num=10, neg_num=10, test_num=0, test_sampling='sequence'):
        """Return positive and negative candidate items for a Steam user.

        Args:
            userid: Steam user id.
            pos_num: Take the last pos_num items from the first 80% of interactions.
            neg_num: Number of randomly sampled non-interacted items.
            test_num: Items from the held-out 20%. -1 returns all; 0 omits test items.
            test_sampling: 'sequence' (chronological) or 'random'.

        Returns:
            (positive_items, negative_items) or (positive_items, negative_items, test_pos_items).
        """
        interacted_items_df = self.interactions_df[self.interactions_df['userid']==userid]
        interacted_items_df = interacted_items_df.merge(self.items_df, how='inner', on=['itemid'])
        

        sorted_interacted_items_df = interacted_items_df.sort_values(by='review_date')
        sorted_interacted_items_df.set_index('interactionid', inplace=True)
        sorted_interacted_items = sorted_interacted_items_df.to_dict(orient='index')
        negative_items_df = self.items_df[~self.items_df['itemid'].isin(sorted_interacted_items.keys())]
        
        negative_items_df.set_index('itemid', inplace=True)
        negative_items = negative_items_df.to_dict(orient='index')
        random_negative_keys = np.random.choice(list(negative_items.keys()), len(sorted_interacted_items), replace=0)
        # unrated_idx_p = np.array(negative_items_df['rating_number'])
        # unrated_idx_p = unrated_idx_p/unrated_idx_p.sum()
        # random_negative_keys = np.random.choice(list(negative_items.keys()), len(sorted_interacted_items), p=unrated_idx_p, replace=0)
        negative_items = [negative_items[key] for key in random_negative_keys][:neg_num]
        
        
        # first 80% for profile generation; last 20% available for test evaluation
        generation_count = int(len(sorted_interacted_items)*0.8)
        positive_items = list(sorted_interacted_items.values())[:generation_count]
        if pos_num > 0:
            positive_items = positive_items[-pos_num:]

        if test_num == -1:
            test_pos_items = list(sorted_interacted_items.values())[generation_count:]
            return positive_items, negative_items, test_pos_items
        if test_num > 0:
            if 'seq' in test_sampling:
                test_pos_items = list(sorted_interacted_items.values())[generation_count:generation_count + test_num]
            else:
                test_pos_items = random.sample(
                    list(sorted_interacted_items.values())[generation_count:],
                    min(test_num, len(sorted_interacted_items) - generation_count),
                )
            return positive_items, negative_items, test_pos_items
        return positive_items, negative_items

    def get_demographic_information(self, userid, activity=False):
        users_dict = self.users_df.set_index('userid').to_dict(orient='index')
        if activity and 'activity' in users_dict[userid].keys():
            return users_dict[userid]['activity']
        return ''
