library(tidyverse)
library(rstan)
library(caret)
library(tidytext)
library(hdf5r)
library(transport)
library(matrixStats)
library(loo)
library(xtable)

#Analysis starts here
KL_calc <- function(p,q){
  idx <- which(p!=0)
  sum(p[idx]*log(p[idx])) - sum(p[idx]*log(q[idx]))
}
source("Aggregation_data_short_log.R")
N_samples = 20000
#Descriptives

#Plots Run time vs. Sequence length
data <- read.csv(file = "../../Data/Aggregation_Modelfits/business_long_seq.csv")
load(file="../../Data/Aggregation_Modelfits/short_seq_ids.R")
binary_label <- function(x){
  scales::math_format(expr = 2^.x)(x)
}
data %>% mutate(binary_count = factor(ceiling(log2(review_count)))) %>%
  filter(!business_id %in% short_seq_ids) %>%
  ggplot(aes(x = binary_count, y = run_time)) + 
  #geom_point()+
  geom_boxplot()+
  theme_classic() + 
  labs(x = "Number of ratings", 
       y = "Runtime in seconds") + 
  scale_x_discrete(labels = binary_label)+
  theme(text = element_text(size=30)) 
rm(data,short_seq_ids)

#Benchmark performance
mean_error <- list()
#sample mean
i = 0 
for(j in 1:NUM_BUSINESS){
  train_seq = Y[i+1:N[j]]
  test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
  p <- prop.table(table(factor(test_seq,levels=1:5)))
  q <- prop.table(table(factor(train_seq,levels=1:5)))
  test_samples <- sample(test_seq,size=N_samples,replace=TRUE)
  train_samples <- sample(train_seq,size=N_samples,replace=TRUE)
  m <- 0.5 * (p + q)
  mean_error[[j]] = c(abs(mean(test_seq)-mean(train_seq)),
                      abs(mean(test_seq)-mean(train_seq))^2,
                      0.5*(KL_calc(p,m)+KL_calc(q,m)),
                      wasserstein1d(test_samples,train_samples))
  i = i + N[j]
}
#MAE: 0.5234 | RMSE: 0.6745 | JS: 0.100 | Wasserstein: 0.598
mean_res = apply(do.call(rbind,mean_error),2,function(v){mean(v,na.rm = TRUE)})
mean_res[2] = sqrt(mean_res[2])
round(mean_res,3)
#Sliding window
sliding_window_error <- discounted_mean_error  <- list()
i = 0
K = 5
for(j in 1:NUM_BUSINESS){
  avg_error = rep(0,N[j]-K)
  train_seq = Y[i+1:N[j]]
  test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
  p <- prop.table(table(factor(test_seq,levels=1:5)))
  test_samples <- sample(test_seq,size=N_samples,replace=TRUE)
  
  for(l in 1:(N[j]-K)){
    n = 0
    for(t in l:(N[j]-K)){
      avg_error[l] = avg_error[l] + abs(mean(train_seq[(t-l+1):t])- mean(train_seq[t+1:K]))
      n = n +1
    }
    avg_error[l] = avg_error[l]/n
  }
  l_star = which.min(avg_error)
  train_seq = train_seq[(N[j]-l_star+1):N[j]]
  q <- prop.table(table(factor(train_seq,levels=1:5)))
  train_samples <- sample(train_seq,size=N_samples,replace=TRUE)
  m <- 0.5 * (p + q)
  sliding_window_error[[j]] = c(abs(mean(test_seq)-mean(train_seq)),
                                abs(mean(test_seq)-mean(train_seq))^2,
                                0.5*(KL_calc(p,m)+KL_calc(q,m)),
                                wasserstein1d(test_samples,train_samples))
  i = i + N[j]
}
sliding_res = apply(do.call(rbind,sliding_window_error),2,function(v){mean(v,na.rm = TRUE)})
sliding_res[2] = sqrt(sliding_res[2])
#0.54 0.71 0.12 0.63
round(sliding_res,3)
#Bayes Average
stan_data = list(NUM_BUSINESS = NUM_BUSINESS,
                  TEST_NUM = TEST_NUM,
                  NUM_REVIEWS = NUM_REVIEWS,
                  N = N ,
                  Y = Y)

bayes_average <- stan(file = "Stan Code/Baselines/bayes_average.stan", 
                      data = stan_data, 
                      chains = 2, 
                      iter = 2000, 
                      refresh = 10)

bayesian_mean_error <- list()
samples <- extract(bayes_average,"pi")
for(j in 1:NUM_BUSINESS){
  test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
  p <- prop.table(table(factor(test_seq,levels=1:5)))
  test_samples <- sample(test_seq,size=N_samples,replace=TRUE)
  q <- apply(samples$pi[,j,],2,mean)
  train_samples = sample(1:5,N_samples,replace = TRUE,prob=q)
  m <- 0.5 * (p + q)
  bayesian_mean_error[[j]] = c(abs(mean(test_seq)-(q %*% 1:5)),
                                abs(mean(test_seq)-(q %*% 1:5))^2,
                                0.5*(KL_calc(p,m)+KL_calc(q,m)),
                                wasserstein1d(test_samples,train_samples))
}
bayes_res = apply(do.call(rbind,bayesian_mean_error),2,function(v){mean(v,na.rm = TRUE)})
bayes_res[2] = sqrt(bayes_res[2])
#0.521 0.645 0.097 0.596
round(bayes_res,3)
#Discounted benchmark
discounted_mean_error  <- list()
stan_data = list(NUM_BUSINESS = NUM_BUSINESS,
                 TEST_NUM = TEST_NUM,
                 NUM_REVIEWS = NUM_REVIEWS,
                 N = N ,
                 Y = Y)

discounted_benchmark <- stan_model(file = "Stan Code/discounted_benchmark.stan")
stan_data$M = 6
map = optimizing(discounted_benchmark,data = stan_data,verbose=TRUE,as_vector=FALSE)
i = 0 
for(j in 1:NUM_BUSINESS){
  test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
  p <- prop.table(table(factor(test_seq,levels=1:5)))
  test_samples <- sample(test_seq,size=N_samples,replace=TRUE)
  train_seq = Y[i+1:N[j]]
  rho = map$par$rho[j]
  v = sapply(1:N[j], function(k){exp(-(N[j]-k+1)*rho)/N[j]})
  train_samples <- sample(train_seq,size=N_samples,replace=TRUE,prob = v)
  q = prop.table(table(factor(train_samples,levels=1:5)))
  m <- 0.5 * (p + q)
  discounted_mean_error[[j]] = c(abs(mean(test_seq)-map$par$agg_rat[j]),
                    abs(mean(test_seq)-map$par$agg_rat[j])^2,
                    0.5*(KL_calc(p,m)+KL_calc(q,m)),
                    wasserstein1d(test_samples,train_samples))
  i = i + N[j]
}
discounted_res = apply(do.call(rbind,discounted_mean_error),2,function(v){mean(v,na.rm = TRUE)})
discounted_res[2] = sqrt(discounted_res[2])
round(discounted_res,3)
#Performance of LGPM Models
sm_df = do.call(rbind,mean_error)
path = "../../Data/Aggregation_Modelfits/model_fit_lgpm_"
models = c("basic","mean","cov_inv_logit","full_inv_logit")
results = list()
stars = matrix(rep(NA,2*length(models)),ncol=2)
for(k in 1:length(models)){
  lgpm_error <- list()
  file.h5 <- H5File$new(paste0(path,models[k]), mode="r+")
  
  samples_ds <- file.h5[["parameters/predictions"]]
  ## note that for now tables in HDF5 are 1-dimensional, not 2-dimensional
  samples <- samples_ds[,]
  predictions = apply(samples,1,mean)
  
  samples_ds <- file.h5[["parameters/rng_ratings"]]
  ## note that for now tables in HDF5 are 1-dimensional, not 2-dimensional
  samples <- samples_ds[,]
  
  file.h5$close() 
  for(j in 1:NUM_BUSINESS){
    test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
    p <- prop.table(table(factor(test_seq,levels=1:5)))
    test_samples <- sample(test_seq,size=N_samples,replace=TRUE)
    
    q <- prop.table(table(factor(samples[(j-1)*TEST_NUM+1:TEST_NUM,],levels = 1:5)))
    train_samples <- sample(samples[(j-1)*TEST_NUM+1:TEST_NUM,],size=N_samples,replace=TRUE)
    m <- 0.5 * (p + q)
    lgpm_error[[j]] = c(abs(mean(test_seq)-mean(predictions[(j-1)*TEST_NUM+1:TEST_NUM])),
                              abs(mean(test_seq)-mean(predictions[(j-1)*TEST_NUM+1:TEST_NUM]))^2,
                              0.5*(KL_calc(p,m)+KL_calc(q,m)),
                              wasserstein1d(test_samples,train_samples))
  }
  lgpm_df = do.call(rbind,lgpm_error)
  res = apply(lgpm_df,2,function(v){mean(v,na.rm = TRUE)})
  for(m in 1:2){
    W = wilcox.test(x = lgpm_df[,m],y=sm_df[,m],paired = TRUE)
    stars[k,m] = ifelse(W$p.value>0.05,0,ifelse(W$p.value>0.01,1,ifelse(W$p.value>0.001,2,3)))
  }
  res[2] = sqrt(res[2])
  results[[k]] = res
}



tbl = round(do.call(rbind,results),3)
rownames(tbl) = models
for(k in 1:length(models)){
  print(round(100*(1-do.call(rbind,results)[k,]/mean_res),3))
}
# Section 5.2 
#Comparison of model fits
N_samples = 2000
source("Aggregation_data_short_log.R")
bayes_R_2 <- function(fitted_Y,Y){
  e <- -1 * sweep(fitted_Y, 2, Y)
  var_ypred <- apply(fitted_Y, 1, var)
  var_e <- apply(e, 1, var)
  median(var_ypred / (var_ypred + var_e))
}
path = "../../Data/Aggregation_Modelfits/model_fit_lgpm_"
models = c("basic","mean","cov_inv_logit","cov_inv_logit_adj_new","full_inv_logit","full_inv_logit_adj_new")
results = list()
for(k in 1:length(models)){
  model_fit <- rep(NA,3)
  file.h5 <- H5File$new(paste0(path,models[k]), mode="r+")
  samples_ds <- file.h5[["parameters/log_lik"]]
  ## note that for now tables in HDF5 are 1-dimensional, not 2-dimensional
  ll <- samples_ds[,]
  
  samples_ds <- file.h5[["parameters/fitted_values"]]
  ## note that for now tables in HDF5 are 1-dimensional, not 2-dimensional
  fitted_Ratings <- samples_ds[,]
  
  file.h5$close()
  model_fit[1] <- -2*sum(apply(ll,1,logSumExp)-log(2000))
  model_fit[2] <- waic(t(ll))$estimates["waic","Estimate"]
  rel_n_eff <- relative_eff(t(ll),chain_id=rep(1:2,each=N_samples/2))
  loo_obj = loo(t(ll),r_eff = rel_n_eff,cores=2)
  model_fit[3] <- loo_obj$estimates["looic","Estimate"]
  model_fit[4] <- bayes_R_2(t(fitted_Ratings),Y)
  results[[length(results)+1]] <- model_fit
}


tbl = do.call(cbind,results)
tbl[4,] = tbl[4,]*100
rownames(tbl) = c("Lpd","WAIC","LOOIC","Bayesian $R^2$")


print.xtable(xtable(tbl,digits = 2) , 
             sanitize.text.function = identity , 
             include.rownames = TRUE , 
             include.colnames = FALSE ,
             type = "latex" , 
             hline.after = NULL,
             booktabs = TRUE , 
             floating = FALSE ,
             only.contents = TRUE,
             file="../../Doc/ISR  - Rating Aggregation/lgpm_information_criteria.tex")

#5.3 Coefficient plot/table
source("Aggregation_data_short_log.R")
path = "../../Data/Aggregation_Modelfits/model_fit_lgpm_"
models = c("basic","mean","cov_inv_logit","cov_inv_logit_adj_new","full_inv_logit","full_inv_logit_adj_new")
model_name = c("Basic","Mean-only LGPM","Covariance-only LGPM a","Covariance-only LGPM b","Full LGPM a","Full LGPM b")
X_MEAN = Q_MEAN %*% R_MEAN
ranges = matrix(c(0.025,0.975,0.005,0.995,0.0005,0.9995),ncol=2,byrow = TRUE)
mean_sd = apply(X_MEAN,2,sd)
results = coef_tbl = stars_l =  list()
rm(Q_COV,Q_MEAN,R_COV,R_MEAN,X_OOS_COV,X_OOS_MEAN,maxDiff,minDiff,N,NUM_BUSINESS,NUM_COV,NUM_MEAN,NUM_REVIEWS,TEST_NUM,X,X_test,Y,Y_test)
ncov = 9
for(k in 1:length(models)){
  coef_vec = rep(NA,2*ncov)
  stars = rep("",ncov)
  if(models[k] == "mean"){
    file.h5 <- H5File$new(paste0(path,models[k]), mode="r+")
    samples_ds <- file.h5[["parameters/omega"]]
    omega = samples_ds[,]
    file.h5$close()
  }else if(models[k] == "cov_inv_logit"||models[k] == "cov_inv_logit_adj_new"){
    file.h5 <- H5File$new(paste0(path,models[k]), mode="r+")
    samples_ds <- file.h5[["parameters/theta"]]
    theta = samples_ds[,]
    file.h5$close()
  }else if(models[k] == "basic"){
    print("No parameter")
  }else{
    file.h5 <- H5File$new(paste0(path,models[k]), mode="r+")
    samples_ds <- file.h5[["parameters/omega"]]
    omega = samples_ds[,]
    samples_ds <- file.h5[["parameters/theta"]]
    theta = samples_ds[,]
    file.h5$close()
  }
  if(exists("omega")){
    PARAMETER = c("Review Sentiment","Mean Rating Score per User")
    TYPE = rep("Valence heterogeneity",length(PARAMETER))
    MODEL = rep(model_name[k],length(PARAMETER))
    MEAN = apply(omega,1,mean)*mean_sd
    SD = apply(omega,1,sd)*mean_sd
    q = apply(omega,1,function(v){quantile(v,c(0.025,0.975))})
    coef_vec[c(1,3)] = MEAN
    coef_vec[c(2,4)] = SD
    LOWER = q[1,]*mean_sd
    UPPER = q[2,]*mean_sd
    t_stars = rep("",2)
    for(l in 1:3){
      q = apply(omega,1,function(v){quantile(v,ranges[l,])})
      t_stars[which(sign(q[1,])==sign(q[2,]))] = paste0(rep("*",l),collapse = "")
    }
    results[[length(results)+1]] = data.frame(MODEL,TYPE,PARAMETER,MEAN,SD,LOWER,UPPER,stringsAsFactors = FALSE)
    stars[c(1,2)] = t_stars
    rm(omega,t_stars)
  }
  if(exists("theta")){
    idx = seq(from=5,to=17,by=2)
    if(length(grep("adj",models[k]))>0){
      source("Aggregation_data_short_log_adj.R")
      idx = idx[-4]
      X_COV = Q_COV %*% R_COV
      cov_sd  = apply(X_COV,2,sd)
      cov_sd[c(3,5)] <- 1
      PARAMETER = c("Review Length","Linguistic Modality",
                    "Temporal Contiguity", "Rating helpfulness", 
                    "Elite Status","Time on Yelp")
      rm(Q_COV,Q_MEAN,R_COV,R_MEAN,X_OOS_COV,X_OOS_MEAN,maxDiff,minDiff,N,NUM_BUSINESS,NUM_COV,NUM_MEAN,NUM_REVIEWS,TEST_NUM,X,X_test,Y,Y_test)
    }else{
      source("Aggregation_data_short_log.R")
      idx = idx[-5]
      X_COV = Q_COV %*% R_COV
      cov_sd  = apply(X_COV,2,sd)
      cov_sd[c(3,5)] <- 1
      PARAMETER = c("Review Length","Linguistic Modality",
                    "Temporal Contiguity", "Rater helpfulness", 
                    "Elite Status","Time on Yelp")
      rm(Q_COV,Q_MEAN,R_COV,R_MEAN,X_OOS_COV,X_OOS_MEAN,maxDiff,minDiff,N,NUM_BUSINESS,NUM_COV,NUM_MEAN,NUM_REVIEWS,TEST_NUM,X,X_test,Y,Y_test)
    }
    TYPE = rep("Credibility",length(PARAMETER))
    MODEL = rep(model_name[k],length(PARAMETER))
    MEAN = apply(theta,1,mean)*cov_sd
    SD = apply(theta,1,sd)*cov_sd
    q = apply(theta,1,function(v){quantile(v,c(0.025,0.975))})
    coef_vec[idx] = MEAN
    coef_vec[idx+1] = SD
    LOWER = q[1,]*cov_sd
    UPPER = q[2,]*cov_sd
    t_stars = rep("",6)
    for(l in 1:3){
      q = apply(theta,1,function(v){quantile(v,ranges[l,])})
      t_stars[which(sign(q[1,])==sign(q[2,]))] = paste0(rep("*",l),collapse = "")
    }
    if(length(grep("adj",models[k]))>0){
      stars[c(3,4,5,7,8,9)] = t_stars  
    }else{
      stars[c(3,4,5,6,8,9)] = t_stars 
    }
    results[[length(results)+1]] = data.frame(MODEL,TYPE,PARAMETER,MEAN,SD,LOWER,UPPER,stringsAsFactors = FALSE)
    rm(theta,t_stars)
  }
  coef_tbl[[length(coef_tbl)+1]] = coef_vec
  stars_l[[length(stars_l)+1]] = stars
}
df = do.call(rbind,results)
df$PARAMETER = as.factor(df$PARAMETER)
df$MODEL= as.factor(df$MODEL)
df$TYPE= as.factor(df$TYPE)
coef_df = do.call(cbind,coef_tbl)
rownames(coef_df) = c("Review sentiment",NA,
                      "Mean rating score per user",NA,
                      "Review Length",NA,
                      "Linguistic Modality",NA,
                      "Temporal Contiguity",NA,
                      "Rater helpfulness", NA,
                      "Rating helpfulness", NA,
                      "Elite Status",NA,
                      "Time on Yelp",NA)


coef_df  = formatC(coef_df,digits = 4,format = "f")
coef_df[seq(from=2,to=18,by=2),] = paste0("(",coef_df[seq(from=2,to=18,by=2),],")")

coef_df[seq(from=1,to=17,by=2),] = paste0("$",coef_df[seq(from=1,to=17,by=2),],"^{",do.call(cbind,stars_l),"}$")

print.xtable(xtable(coef_df) , 
             sanitize.text.function = identity , 
             include.rownames = TRUE , 
             include.colnames = FALSE ,
             type = "latex" , 
             hline.after = NULL,
             booktabs = TRUE , 
             floating = FALSE ,
             only.contents = TRUE,
             file="../../Doc/ISR  - Rating Aggregation/lgpm_coefficients.tex")



p <- ggplot(transform(df, 
                 TYPE = factor(TYPE,levels = c('Valence heterogeneity','Credibility')),
                 MODEL = factor(MODEL,levels = model_name)), 
       aes(x = PARAMETER, y = MEAN)) + 
  geom_hline(yintercept = 0, lty = 2, lwd = 0.5,color="grey") + 
  geom_errorbar(aes(ymin = LOWER , 
                    ymax = UPPER , 
                    color = MODEL),
                lwd=1, 
                width=0,
                position = position_dodge(width = 0.5)) + 
  geom_point(aes(color = MODEL),
             position = position_dodge(width = 0.5),
             shape = 21,fill="white",
             size = 2.5) +
  scale_color_brewer(type="qual",palette = 6)+
  ylab("Coefficient") + 
  coord_flip()+ 
  facet_wrap(~TYPE, scales = "free",nrow=2,drop = TRUE,shrink = FALSE)+
  theme_bw() + 
  theme(legend.title = element_blank(),
        text=element_text(size=20), 
        axis.title.y = element_blank(),
        legend.position="top",
        strip.text = element_text(face="bold"),
        strip.background = element_blank())

gp <- ggplotGrob(p)

# optional: take a look at the grob object's layout

facet.rows <- gp$layout$t[grepl("panel", gp$layout$name)]
x.var <- sapply(ggplot_build(p)$layout$panel_scales_x,
                function(l) length(l$range$range))
gp$heights[facet.rows] <- gp$heights[facet.rows] * x.var
grid::grid.draw(gp)

rm(list=ls())

#Robustness checks 

#Performance for different number of reviews
source("Aggregation_data_short_log.R")
N_samples = 20000
mean_error <- lgpm_error <- list()
path = "../../Data/Aggregation_Modelfits/model_fit_lgpm_basic"
file.h5 <- H5File$new(path, mode="r+")

samples_ds <- file.h5[["parameters/predictions"]]
## note that for now tables in HDF5 are 1-dimensional, not 2-dimensional
samples <- samples_ds[,]
predictions = apply(samples,1,mean)

file.h5$close() 
#sample mean and basic for first 400 reviews
i = 0 
for(j in 1:NUM_BUSINESS){
  train_seq = Y[i+1:N[j]]
  test_seq = Y_test[(j-1)*TEST_NUM+1:TEST_NUM]
  mean_error[[j]] = abs(mean(test_seq)-mean(train_seq))
  lgpm_error[[j]] = abs(mean(test_seq)-mean(predictions[(j-1)*TEST_NUM+1:TEST_NUM]))
  i = i + N[j]
}
load("../../Data/Aggregation_Modelfits/short_seq_ids.R")
df = data.frame(ID = short_seq_ids,Length = N, Mean = unlist(mean_error),LGPM = unlist(lgpm_error),stringsAsFactors = FALSE)

df_long = read_csv(file="../../Data/Aggregation_Modelfits/error_comparison.csv") %>% select(-X1)

df_long$LGPM = as.numeric(gsub("\\[|\\]", "",df_long$LGPM))

df_long$ID = gsub("\n","",stringr::str_extract(df_long$ID,"[:graph:]*\n"))
df_long$Length = df_long$Length-10  
df = rbind(df,df_long) %>% distinct(ID,.keep_all = TRUE)

breaks =  c(10,25,50,100,150,200,500,Inf)
tbl = list()
for(i in 1:(length(breaks)-1)){
  results_vec = rep(NA,6)
  lower = breaks[i]
  upper = breaks[i+1]
  results_vec[1] = paste0("[",lower,",",upper,")")
  df_temp = df %>% filter(Length >= lower,Length<upper)
  results_vec[2] = nrow(df_temp)
  results_vec[3] = mean(df_temp$Mean)
  results_vec[4] = sqrt(mean(df_temp$Mean^2))
  results_vec[5] = mean(df_temp$LGPM)
  results_vec[6] = sqrt(mean(df_temp$LGPM^2))
  tbl[[length(tbl)+1]] = results_vec
}
tbl = do.call(rbind,tbl)

colnames(tbl) = c("Range","Number restaurants","MAE - SM","RMSE - SM","MAE - GP","RMSE - GP")
print.xtable(xtable(tbl,digits = 4) , 
             sanitize.text.function = identity , 
             include.rownames = FALSE , 
             include.colnames = FALSE ,
             type = "latex" , 
             hline.after = NULL,
             booktabs = TRUE , 
             floating = FALSE ,
             only.contents = TRUE,
             file="../../Doc/ISR  - Rating Aggregation/bracket_performance.tex")

#Descriptives table
#DO NOT RUN
load(file="../../Data/Aggregation_Modelfits/business_ids.R")


reviews = read_csv(file = "../../Data/yelp_review.csv", progress = TRUE) %>% 
  filter(business_id %in% business_ids)


timecontiguity_keywords = c("today", "this morning", "just got back", "tonight", "yesterday", "last night")
StrongModal = tolower(c("ALWAYS","BEST","CLEARLY","DEFINITELY","DEFINITIVELY","HIGHEST","LOWEST","MUST","NEVER", "STRONGLY",
                        "UNAMBIGUOUSLY","UNCOMPROMISING","UNDISPUTED","UNDOUBTEDLY","UNEQUIVOCAL", "UNEQUIVOCALLY","UNPARALLELED", "UNSURPASSED","WILL"))
WeakModal = tolower(c("ALMOST","APPARENTLY","APPEARED","APPEARING","APPEARS", "CONCEIVABLE","COULD","DEPEND","DEPENDED","DEPENDING",
                      "DEPENDS","MAY","MAYBE","MIGHT","NEARLY","OCCASIONALLY","PERHAPS","POSSIBLE","POSSIBLY","SELDOM","SELDOMLY",
                      "SOMETIMES","SOMEWHAT","SUGGEST","SUGGESTS","UNCERTAIN","UNCERTAINLY"))

DictionaryModal = SentimentDictionaryBinary(StrongModal,WeakModal)
rules_sentiment = list("SentimentQDAP" = list(ruleSentiment, loadDictionaryQDAP()),
                       "LinguisticModality" = list(ruleSentiment,DictionaryModal))

rm(StrongModal,WeakModal)

SentimentQDAP <- TempContiguity <- LinguisticModality <- ReviewLength <- c()


for(n in 1:length(business_ids)) {
  cat(paste0(n, "/", length(business_ids), "\n"))
  
  curr_text <- reviews %>% 
    filter(business_id == business_ids[n]) 
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
  LinguisticModality <- c(LinguisticModality,sentiment$LinguisticModality)
  ReviewLength <- c(ReviewLength,unlist(countWords(curr_text$text, removeStopwords = F)))
  rm(sentiment)
}
# Replace NA values by 0

ReviewLength[is.na(ReviewLength)] <- 0
SentimentQDAP[is.na(SentimentQDAP)] <- 0
LinguisticModality[is.na(LinguisticModality)] <- 0
rm(curr_text, n,j)
#Checks
length(ReviewLength) == length(SentimentQDAP)

length(ReviewLength) == length(LinguisticModality)
length(ReviewLength) == length(TempContiguity)
length(ReviewLength) == nrow(reviews)
reviews = reviews %>% add_column(ReviewLength,SentimentQDAP,LinguisticModality,TempContiguity)
rm(rules_sentiment,TempContiguity,timecontiguity_keywords,SentimentQDAP,LinguisticModality,ReviewLength,DictionaryModal)

#Load user data


user = read_csv(file = "../../Data/yelp_user.csv", progress = TRUE)
user = user %>% select(user_id, review_count, elite, average_stars, useful,yelping_since) %>%
  mutate(useful_reviewer = useful) %>% select(-useful)
reviews = reviews %>%
  left_join(user,by="user_id") %>%
  rowwise() %>%
  mutate(elite_status = ifelse(elite == "None",0,1),
         time_on_yelp = as.numeric(difftime(date,yelping_since,units = "days"))) %>% 
  select(-yelping_since,-funny,-cool,-elite) 
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
         l_useful_review = log1p(useful),
         l_average_useful = l_useful_reviewer-l_review_count)
df <- reviews %>% select(stars,SentimentQDAP,adj_user_average,TempContiguity,ReviewLength,LinguisticModality,
                         time_on_yelp,elite_status,l_average_useful,l_useful_review)
df$time_on_yelp <- df$time_on_yelp/365
df$useful = exp(df$l_average_useful)
df$review_useful = exp(df$l_useful_review)-1
round(apply(df,2,function(v){mean(v,na.rm = TRUE)}),2)
round(apply(df,2,function(v){sd(v,na.rm = TRUE)}),2)


rm(list=ls())








