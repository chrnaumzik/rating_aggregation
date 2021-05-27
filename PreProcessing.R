library(tidyverse)
library(rstan)
library(caret)
library(SentimentAnalysis)
library(tidytext)



TARGET_CITY = "Phoenix"
MIN_REVIEWS = 20
MAX_REVIEWS = 200
NUM_BUSINESS = 400
TEST_NUM = 10

business = read_csv(file="../../Data/yelp_business.csv", progress = TRUE)

business = business %>% filter(city == TARGET_CITY) %>% 
                        separate(categories,";",into = paste0("V_",1:25),extra="drop",fill="right") %>% 
                        filter_at(vars(starts_with("V_")), any_vars(.=="Restaurants")) %>% 
  select_at(vars(-starts_with("V_")))

reviews = read_csv(file = "../../Data/yelp_review.csv", progress = TRUE) %>% 
  filter(business_id %in% business$business_id) %>% mutate(Year=format(date,"%Y")) 
age_df = reviews %>% 
  group_by(business_id) %>% 
  summarise(MIN_DATE = min(date), MAX_DATE = max(date),FIRST_YEAR = min(Year),FREQ = n()) %>% 
  mutate(Age = as.numeric(difftime(MAX_DATE,MIN_DATE,units = "days"))) %>% 
  select(business_id,Age,FIRST_YEAR,FREQ)
#Age >= MIN_AGE,
business = business %>% left_join(age_df,by = "business_id") %>% filter(FIRST_YEAR>=2010,FREQ>=MIN_REVIEWS,FREQ<=MAX_REVIEWS)
reviews = reviews %>% filter(business_id %in% business$business_id)
rm(age_df)


timecontiguity_keywords = c("today", "this morning", "just got back", "tonight", "yesterday", "last night")
StrongModal = tolower(c("ALWAYS","BEST","CLEARLY","DEFINITELY","DEFINITIVELY","HIGHEST","LOWEST","MUST","NEVER", "STRONGLY",
                         "UNAMBIGUOUSLY","UNCOMPROMISING","UNDISPUTED","UNDOUBTEDLY","UNEQUIVOCAL", "UNEQUIVOCALLY","UNPARALLELED", "UNSURPASSED","WILL"))
WeakModal = tolower(c("ALMOST","APPARENTLY","APPEARED","APPEARING","APPEARS", "CONCEIVABLE","COULD","DEPEND","DEPENDED","DEPENDING",
                       "DEPENDS","MAY","MAYBE","MIGHT","NEARLY","OCCASIONALLY","PERHAPS","POSSIBLE","POSSIBLY","SELDOM","SELDOMLY",
                       "SOMETIMES","SOMEWHAT","SUGGEST","SUGGESTS","UNCERTAIN","UNCERTAINLY"))

afinn_list = get_sentiments("afinn")

DictionaryAFINN = SentimentDictionaryWeighted(afinn_list$word,afinn_list$value)
DictionaryModal = SentimentDictionaryBinary(StrongModal,WeakModal)
rules_sentiment = list("SentimentQDAP" = list(ruleSentiment, loadDictionaryQDAP()) ,
                        "SentimentAFINN" = list(ruleLinearModel,DictionaryAFINN),
                        "LinguisticModality" = list(ruleSentiment,DictionaryModal))

rm(StrongModal,WeakModal)

SentimentQDAP <- SentimentAFINN <- TempContiguity <- LinguisticModality <- ReviewLength <- c()


for(n in 1:nrow(business)) {
  cat(paste0(n, "/", nrow(business), "\n"))
  
  curr_text <- reviews %>% 
    filter(business_id == business$business_id[n]) 
  #Removing non-latin characters from review text as these cause error with tolower
  curr_text$text <- gsub("[^\U0020-\U007F]", "", curr_text$text)
  
  time_temp <- rep(0,nrow(curr_text))
  for(j in 1:length(timecontiguity_keywords)){
    time_temp[grep(timecontiguity_keywords[j],curr_text$text,ignore.case = TRUE)] <- 1
  }
  TempContiguity <- c(TempContiguity,time_temp)
  rm(time_temp)
  
  sentiment <- analyzeSentiment(tolower(curr_text$text), rules = rules_sentiment)
  SentimentQDAP <- c(SentimentQDAP,sentiment$SentimentQDAP)
  SentimentAFINN <- c(SentimentAFINN,sentiment$SentimentAFINN)
  LinguisticModality <- c(LinguisticModality,sentiment$LinguisticModality)
  ReviewLength <- c(ReviewLength,unlist(countWords(curr_text$text, removeStopwords = F)))
  rm(sentiment)
}
# Replace NA values by 0

ReviewLength[is.na(ReviewLength)] <- 0
SentimentQDAP[is.na(SentimentQDAP)] <- 0
SentimentAFINN[is.na(SentimentAFINN)] <- 0
LinguisticModality[is.na(LinguisticModality)] <- 0
rm(curr_text, n,j)
#Checks
length(ReviewLength) == length(SentimentQDAP)
length(ReviewLength) == length(SentimentAFINN)
length(ReviewLength) == length(LinguisticModality)
length(ReviewLength) == length(TempContiguity)
length(ReviewLength) == nrow(reviews)
reviews = reviews %>% add_column(ReviewLength,SentimentAFINN,SentimentQDAP,LinguisticModality,TempContiguity)
rm(rules_sentiment,TempContiguity,timecontiguity_keywords,SentimentAFINN,SentimentQDAP,LinguisticModality,ReviewLength,DictionaryModal,DictionaryAFINN,afinn_list)

#Load user data


user = read_csv(file = "../../Data/yelp_user.csv", progress = TRUE)
user = user %>% select(user_id, review_count, elite, average_stars, useful,yelping_since,friends) %>%
  mutate(useful_reviewer = useful) %>% select(-useful)
reviews = reviews %>%
  left_join(user,by="user_id") %>%
  rowwise() %>%
  mutate(elite_status = ifelse(elite == "None",0,1),
         time_on_yelp = as.numeric(difftime(date,yelping_since,units = "days")),
         no_friends = length(unlist(strsplit(friends,",")))) %>% 
  select(-yelping_since,-funny,-cool,-elite,-friends) 
rm(user)




reviews <- reviews %>% 
  rowwise() %>% 
  mutate(adj_user_average = ifelse(review_count<=1,0,(review_count*average_stars-stars)/(review_count-1))) %>%
  mutate(adj_user_average = ifelse(adj_user_average>5,average_stars,ifelse(adj_user_average<0,average_stars,adj_user_average))) %>%
  mutate(l_review_count = log1p(review_count),
         l_time_on_yelp = log1p(time_on_yelp),
         l_useful = log1p(useful),
         l_review_length = log1p(ReviewLength),
         l_useful_reviewer= log1p(useful_reviewer),
         l_average_useful = l_useful_reviewer-l_review_count)






set.seed(89909)
idx = sample(1:nrow(business),size = NUM_BUSINESS,replace = FALSE)
i = 1
Y = Y_test = X = X_test = N = minDiff = maxDiff = c()
X_MEAN_OOS = X_COV_OOS = list()
X_MEAN_L = X_COV_L = list()
mean_vec = c("SentimentQDAP","adj_user_average")
cov_vec = c("l_review_length","LinguisticModality","TempContiguity", "l_useful","elite_status","l_time_on_yelp")
for(it in idx){
  print(paste("============",round(i/length(idx)*100,1),"% ============"))
  review_df = reviews %>% 
    filter(business_id == business$business_id[it]) %>% 
    select(date, stars, l_useful, l_review_length, 
           SentimentQDAP, 
           LinguisticModality, 
           TempContiguity, 
           l_time_on_yelp,
           adj_user_average,elite_status) %>%
    arrange(date)
  y = review_df$stars
  x = as.numeric(difftime(review_df$date,review_df$date[1],units="days"))
  n = length(y) - TEST_NUM
  x = scale(x)
  x = jitter(x,0.001)
  Y = c(Y,y[1:n])
  Y_test = c(Y_test,y[n+1:TEST_NUM])
  X = c(X,x[1:n])
  X_test = c(X_test,x[n+1:TEST_NUM])
  N = c(N,n)
  minDiff = c(minDiff,1.0/attr(x,"scaled:scale"))
  maxDiff = c(maxDiff,max(x[1:n]) - min(x[1:n]))
  X_MEAN_L[[i]] = review_df[1:n,mean_vec]
  X_COV_L[[i]] =   review_df[1:n,cov_vec]
  X_MEAN_OOS[[i]] = rep(1,TEST_NUM) %*% t.default(apply(X_MEAN_L[[i]],2,mean))
  X_COV_OOS[[i]] =   rep(1,TEST_NUM) %*% t.default(apply(X_COV_L[[i]],2,mean))
  i = i + 1
}
X_MEAN = do.call(rbind,X_MEAN_L)
colnames(X_MEAN) = mean_vec
preProc_mean = preProcess(X_MEAN,c("center"))
X_MEAN = predict(preProc_mean,X_MEAN)

X_OOS_MEAN = do.call(rbind,X_MEAN_OOS)
colnames(X_OOS_MEAN) = mean_vec
X_OOS_MEAN = predict(preProc_mean,X_OOS_MEAN)

QR_MEAN = qr(X_MEAN)
Q_MEAN = qr.Q(QR_MEAN)*sqrt(nrow(X_MEAN)-1)
R_MEAN = qr.R(QR_MEAN)/sqrt(nrow(X_MEAN)-1)
NUM_MEAN = ncol(Q_MEAN)


X_COV = do.call(rbind,X_COV_L)
colnames(X_COV) = cov_vec
preProc_cov = preProcess(X_COV,c("center"))
X_COV = predict(preProc_cov,X_COV)

X_OOS_COV = do.call(rbind,X_COV_OOS)
colnames(X_OOS_COV) = cov_vec
X_OOS_COV = predict(preProc_cov,X_OOS_COV)

QR_COV = qr(X_COV)
Q_COV = qr.Q(QR_COV)*sqrt(nrow(X_COV)-1)
R_COV = qr.R(QR_COV)/sqrt(nrow(X_COV)-1)
NUM_COV = ncol(Q_COV)

NUM_REVIEWS = length(Y)#
#Checks for data consistency
length(X_test) == length(Y_test)
length(X_test) == NUM_BUSINESS*TEST_NUM
length(X) == NUM_REVIEWS
length(X) == sum(N)
length(minDiff) == length(maxDiff)
length(minDiff) == length(N)
nrow(Q_MEAN) == nrow(Q_COV)
nrow(Q_MEAN) == NUM_REVIEWS
nrow(X_OOS_COV) == nrow(X_OOS_MEAN)
nrow(X_OOS_COV) == TEST_NUM*NUM_BUSINESS
stan_rdump(list = list("NUM_BUSINESS","NUM_REVIEWS","NUM_COV","NUM_MEAN","TEST_NUM","N","Y", 
                       "Y_test","X","X_test","minDiff","maxDiff","Q_MEAN","R_MEAN","X_OOS_MEAN","Q_COV","R_COV","X_OOS_COV"), 
           file = "Aggregation_data_short_log_adj.R")

#Descriptive tables
short_seq_ids = business$business_id[idx]
save(short_seq_ids ,file="../../Data/Aggregation_Modelfits/short_seq_ids.R")

rm(list=ls())

#Dawid-skene data
library(feather)
load(file="../../Data/Aggregation_Modelfits/short_seq_ids.R")
reviews = read_csv(file = "../../Data/yelp_review.csv", progress = TRUE) %>% 
  filter(business_id %in% short_seq_ids) %>% mutate(Year=format(date,"%Y")) 
business = tibble(business_id = short_seq_ids,RES_ID = 1:length(short_seq_ids))
df = reviews %>% select(business_id,user_id,stars,date) %>% 
  group_by(business_id) %>% 
  arrange(date) %>%
  mutate(id = 1:n()) %>%
  mutate(MAX_id = max(id)) %>%
  mutate(train = ifelse(id<MAX_id-9,1,0)) %>%
  select(-id,-MAX_id) %>%
  ungroup()


users = df %>% select(user_id) %>% distinct(user_id) %>% mutate(USER_ID = 1:n())

df = df %>% left_join(business,by="business_id") %>% left_join(users,by="user_id") 

train_df = df %>% filter(train == 1) %>% arrange(RES_ID) %>% select(-train,-business_id,-user_id,-date)
test_df = df %>% filter(train == 0) %>% arrange(RES_ID) %>% select(-train,-business_id,-user_id,-date)
write_feather(train_df,path = "../../Data/crowdscourcing_data_train.feather")
write_feather(test_df,path = "../../Data/crowdscourcing_data_test.feather")
rm(list=ls())

