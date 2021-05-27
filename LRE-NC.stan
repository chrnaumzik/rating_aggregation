data {
  int<lower=1> J; //number of annotators
  int<lower=2> K; //number of classes
  int<lower=1> N; //number of annotations
  int<lower=1> I; //number of items
  int<lower=1,upper=I> ii[N]; //the item the n-th annotation belongs to
  int<lower=1,upper=J> jj[N]; //the annotator which produced the n-th annotation
  int y[N]; //the class of the n-th annotation
}

transformed data {
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
}

parameters {
  simplex[K] pi;
  
  matrix[I,K-1] theta_raw; 
  matrix<lower=0>[K,K-1] Chi;
  
  matrix[K,K-1] beta_raw[J];
  matrix[K,K-1] zeta;
  matrix<lower=0>[K,K-1] Omega;
}

transformed parameters {
  
  vector[K] log_q_c[I];
  vector[K] log_pi;
  matrix[K,K-1] beta[J];
  matrix[K,K-1] theta[I];
  
  log_pi = log(pi);
  
  for(j in 1:J)
    beta[j] = zeta + Omega .* beta_raw[j]; //non-centered parameterization
  
  for (i in 1:I)
  {
    for(h in 1:K)
    {
      log_q_c[i,h] = log_pi[h];
      
      theta[i,h] = Chi[h] .* theta_raw[i]; //non-centered parameterization
    }
    
  }
  
  for (n in 1:N)
  {
    for (h in 1:K)
    {
      row_vector[K-1] temp = beta[jj[n],h] - theta[ii[n],h];
      row_vector[K] temp_i = append_col(temp, 0); //fix the last category to 0
      
      log_q_c[ii[n],h] = log_q_c[ii[n],h] + temp_i[y[n]] - log_sum_exp(temp_i);
    }
  }
}

model 
{
  pi ~ dirichlet(alpha);
  
  to_vector(zeta) ~ normal(0, 1);
  to_vector(Omega) ~ normal(0, 1);
  
  to_vector(Chi) ~ normal(0, 1);
  
  for(j in 1:J)
    to_vector(beta_raw[j]) ~ normal(0, 1);
  
  for (i in 1:I)
  {
    theta_raw[i] ~ normal(0, 1);
    target += log_sum_exp(log_q_c[i]);
  }
}

generated quantities {
  vector[K] q_z[I]; //the true class distribution of each item
  
  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
