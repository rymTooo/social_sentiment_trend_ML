
import re
import pandas as pd
import  newmm_tokenizer.tokenizer


class Thai_tokenizer():



        def __init__(self) -> None:
                pass

        def process_tweet(self, tweet, keep_whitespace=False):

                tweet = re.sub(r'\$', '', tweet)  # remove stock market tickers like $GE
                tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
                tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # remove hyperlinks
                tweet = re.sub(r'#', '', tweet)  # remove hashtags
                tweet = re.sub(r'\@\w*', '', tweet)  # remove mention
                # tweet = re.sub(r'\d+', '', tweet) # remove number
                tweets_clean = newmm_tokenizer.tokenizer.word_tokenize(tweet, keep_whitespace=keep_whitespace)

                return tweets_clean

                # print(test_sentence)

                # print(process_tweet(test_sentence))


test_sentence = """โครงสร้างเศรษฐกิจของจังหวัดเชียงรายมาจากการเกษตร ป่าไม้ และการประมงเป็นหลัก 
พืชสำคัญทางเศรษฐกิจของจังหวัดเชียงราย ได้แก่ ข้าวจ้าว ข้าวโพดเลี้ยงสัตว์ สัปปะรด มันสัมปะหลัง 
ส้มโอ ลำไย และลิ้นจี่ ซึ่งทั้งคู่เป็นผลไม้สำคัญที่สามารถปลูกได้ในทุกอำเภอของจังหวัด"""

test_sentence = pd.read_csv("model/thai_tweets.csv")
test_sentence = test_sentence['snippet'][1]

new = Thai_tokenizer()
print(new.process_tweet(test_sentence))