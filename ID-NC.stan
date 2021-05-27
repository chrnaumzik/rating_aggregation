data {
  // int<lower=1> J; //number of annotators
  int<lower=2> K; //number of classes
  int<lower=1> N; //number of annotations
  int<lower=1> I; //number of items
  int<lower=1,upper=I> ii[N]; //the item the n-th annotation belongs to
  // int<lower=1,upper=J> jj[N]; //the annotator which produced the n-th annotation
  int y[N]; //the class of the n-th annotation
}

transformed data
{
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
}

parameters {
  simplex[K] pi;
  matrix[I,K-1] theta_raw;
  
  matrix[K,K-1] eta;
  matrix<lower=0>[K,K-1] Chi;
}

transformed parameters {
  
  vector[K] log_q_c[I];
  vector[K] log_pi;
  matrix[K,K-1] theta[I];
  matrix[K,K] theta_norm[I];
  
  log_pi = log(pi);
  
  for (i in 1:I) 
  {
    for(h in 1:K)
    {
      theta[i,h] = eta[h] + Chi[h] .* theta_raw[i]; //non-centered parameterization

      theta_norm[i,h] = append_col(theta[i,h], 0); //fix last category to 0
      theta_norm[i,h] = theta_norm[i,h] - log_sum_exp(theta_norm[i,h]); //cache the log softmax
      
      log_q_c[i,h] = log_pi[h];
    }
  }
  
  for(n in 1:N)
    for(h in 1:K)
      log_q_c[ii[n],h] = log_q_c[ii[n],h] + theta_norm[ii[n],h,y[n]];
}

model {
  
  to_vector(eta) ~ normal(0,1);
  to_vector(Chi) ~ normal(0,1);
  to_vector(theta_raw) ~ normal(0,1);
  
  pi ~ dirichlet(alpha);
  
  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  vector[K] q_z[I]; //the true class distribution of each item
  
  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
