# load the required libararies
library(fpp2)        
library(dplyr)       
library(urca)       
library(stats)       
library(lmtest)      
library(ggplot2)
library(vars)

# PART 1 - DATA LOADING AND INITIAL EXPLORATION

# Load the dataset
nasdaq_data <- read.csv("/Users/priyadarshinibalasundaram/Downloads/nasdq.csv")

# Convert Date column to Date format
nasdaq_data$Date <- as.Date(nasdaq_data$Date)

# Checking the first few rows of the dataset
head(nasdaq_data)

# Summary statistics
summary(nasdaq_data)

# Check for missing values
sum(is.na(nasdaq_data))

# Visualize interest rates over time with autoplot
interest_ts_full <- ts(nasdaq_data$InterestRate, frequency = 365)
autoplot(interest_ts_full) +
  labs(title = "Interest Rate Over Time", 
       x = "Time", 
       y = "Interest Rate") 

# PART 2 - FILTER DATA AND CONVERT TO MONTHLY FREQUENCY

# Filter to last 10 years of data
current_date <- max(nasdaq_data$Date)
start_date <- current_date - 365*10  # Approximate 10 years in days
nasdaq_filtered <- nasdaq_data %>% filter(Date >= start_date)

# Create monthly data by taking end-of-month values
nasdaq_monthly <- nasdaq_filtered %>%
  mutate(year = as.numeric(format(Date, "%Y")),
         month = as.numeric(format(Date, "%m"))) %>%
  group_by(year, month) %>%
  filter(Date == max(Date)) %>%  # Take end-of-month values
  ungroup() %>%
  select(-year, -month) %>%
  arrange(Date)

# Print dimensions of monthly data
dim(nasdaq_monthly) 

# PART 3 - CREATE TIME SERIES OBJECT AND VISUALIZE 

# Create a monthly time series object for interest rates
interest_ts <- ts(nasdaq_monthly$InterestRate, 
                  frequency = 12, 
                  start = c(as.numeric(format(min(nasdaq_monthly$Date), "%Y")), 
                            as.numeric(format(min(nasdaq_monthly$Date), "%m"))))

# Check the structure of the time series object
print(interest_ts)

# Get start and end of the time series
print(paste("Start:", start(interest_ts)))
print(paste("End:", end(interest_ts)))

# Visualize the monthly interest rate time series
autoplot(interest_ts) + 
  ggtitle("Monthly Interest Rate") + 
  xlab("Year") + 
  ylab("Interest Rate") +
  theme_minimal()

# Basic statistics of the time series
print(summary(interest_ts))

# Check for trend and seasonality visually
# Decompose the time series to see components
decomposed <- decompose(interest_ts)
autoplot(decomposed) +
  ggtitle("Decomposition of Interest Rate Time Series")

# Looking at our decomposition graph, we can see clear patterns that guide 
# our model selection. The decomposition shows a dominant trend component with 
# dramatic shifts (especially the 2020 drop and 2022-2024 rise), minimal 
# seasonality, and large spikes in the remainder component during major policy 
# events. These patterns suggest we need models that can handle strong trends 
# and capture external economic influences. ARIMAX is our primary choice because 
# it combines differencing for the trend with external variables to explain those 
# large remainder spikes. We'll compare this against a standard ARIMA to see if 
# those economic variables really help, test Dynamic Regression to focus on how 
# economic factors drive rates, try VAR to capture two-way relationships between
# rates and the economy, and use Simple Exponential Smoothing as a basic benchmark. 
# Each model addresses different aspects we see in the decomposition - the trend, 
# the policy-driven irregularities, and the minimal seasonality.

# Part 4: Stationarity Testing

# Test if the interest rate data is stationary
# We use the ADF test - if the test value is smaller than critical value, it's stationary
adf_test <- ur.df(interest_ts, type = "drift", lags = 1)
summary(adf_test)

# Extract test statistic and critical value at 5%
test_value <- adf_test@teststat[1]
critical_value <- adf_test@cval[1,2]  # 5% significance level

# Print results in simple terms
print(paste("Test value:", round(test_value, 4)))
print(paste("Critical value (5%):", round(critical_value, 4)))

# Determine if stationary
if (test_value > critical_value) {
  print("Result: Data is NOT stationary (test value > critical value)")
  print("We need to difference the data")
  
  # Difference the data (subtract each value from the next)
  interest_diff <- diff(interest_ts)
  
  # Test the differenced data
  adf_diff_test <- ur.df(interest_diff, type = "drift", lags = 1)
  
  # Check if differenced data is stationary
  diff_test_value <- adf_diff_test@teststat[1]
  diff_critical_value <- adf_diff_test@cval[1,2]
  
  print(paste("Differenced test value:", round(diff_test_value, 4)))
  print(paste("Critical value (5%):", round(diff_critical_value, 4)))
  
  if (diff_test_value < diff_critical_value) {
    print("Success! Differenced data IS stationary")
  }
} else {
  print("Result: Data is already stationary")
  interest_diff <- interest_ts
}

# Plot the differenced data
autoplot(interest_diff) + 
  ggtitle("Differenced Interest Rate Series") + 
  xlab("Year") + 
  ylab("Change in Interest Rate")

# Part 5: Split Data and Analyze ACF/PACF

# Split data into training and testing sets (80% train, 20% test)
n <- length(interest_diff)
n_train <- floor(0.8 * n)

# Create training and test sets
train_interest <- window(interest_diff, end = time(interest_diff)[n_train])
test_interest <- window(interest_diff, start = time(interest_diff)[n_train + 1])

# Print the split information
print(paste("Total observations:", n))
print(paste("Training observations:", n_train))
print(paste("Testing observations:", n - n_train))

# Look at ACF and PACF of training data
ggtsdisplay(train_interest, main = "ACF and PACF of Training Data (Differenced)")

# PART 6: MODEL FITTING AND COMPARISON

# 6.1 

# Fit ETS Model as Benchmark
train_original <- window(interest_ts, end = time(interest_ts)[n_train])
test_original <- window(interest_ts, start = time(interest_ts)[n_train + 1])

# ETS automatically selects the best model type
ets_model <- ets(train_original)  # Let the function pick the best model
print("ETS Model Results:")
print(summary(ets_model))
print(paste("ETS Model Selected:", ets_model$method))

# Make forecasts with ETS for the test period
ets_forecast <- forecast(ets_model, h = length(test_original))

# Create a nice plot comparing ETS forecast vs actual
plot(ets_forecast, main = "ETS Forecast vs Actual", 
     xlab = "Time", ylab = "Interest Rate",
     fcol = "blue", col = "blue")
lines(test_original, col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1)

# Check how accurate the ETS forecast was
print("ETS Forecast Accuracy:")
ets_accuracy <- accuracy(ets_forecast, test_original)
print(ets_accuracy)

# 6.2

# Try Different ARIMA Models
arima_110 <- Arima(train_interest, order = c(1,1,0))
arima_011 <- Arima(train_interest, order = c(0,1,1))
arima_111 <- Arima(train_interest, order = c(1,1,1))
arima_212 <- Arima(train_interest, order = c(2,1,2))
arima_112 <- Arima(train_interest, order = c(1,1,2))
arima_211 <- Arima(train_interest, order = c(2,1,1))

# Print summaries to see AICc values
print("ARIMA(1,1,0) Summary:")
summary(arima_110) # -83.71

print("ARIMA(0,1,1) Summary:")
summary(arima_011) # -93.33

print("ARIMA(1,1,1) Summary:")
summary(arima_111) # -95.62 

print("ARIMA(2,1,2) Summary:")
summary(arima_212) # -95.45

print("ARIMA(1,1,2) Summary:")
summary(arima_112) # -97.51

print("ARIMA(2,1,1) Summary:")
summary(arima_211) # -96.18

# Identify best model based on AICc
best_arima <- arima_112  # ARIMA(1,1,2) has lowest AICc
best_model_name <- "ARIMA(1,1,2)"
print(paste("Best manually selected ARIMA model based on AICc:", best_model_name))

# Make forecasts with the best ARIMA model
best_arima_forecast <- forecast(best_arima, h = length(test_interest))

# Plot the forecast
plot(best_arima_forecast, main = paste(best_model_name, "Forecast vs Actual"),
     xlab = "Time", ylab = "Change in Interest Rate")
lines(test_interest, col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1)

# Check the accuracy of the best ARIMA model
print(paste(best_model_name, "Forecast Accuracy:"))
best_arima_accuracy <- accuracy(best_arima_forecast, test_interest)
print(best_arima_accuracy)

# 6.3 
# dynamic reg

# Prepare exogenous variables for ARIMAX models
# We need to create training and test portions for exogenous variables
exog_vars <- cbind(
  ExchangeRate = nasdaq_monthly$ExchangeRate[-1],  # Remove first observation due to differencing
  VIX = nasdaq_monthly$VIX[-1],
  TEDSpread = nasdaq_monthly$TEDSpread[-1],
  EFFR = nasdaq_monthly$EFFR[-1]
)

# Split into training and testing sets
train_xreg <- exog_vars[1:n_train, ]
test_xreg <- exog_vars[(n_train+1):n, ]

# Check dimensions of training and testing sets
print(paste("Training set dimensions:", dim(train_xreg)[1], "x", dim(train_xreg)[2]))
print(paste("Testing set dimensions:", dim(test_xreg)[1], "x", dim(test_xreg)[2]))

# Fit ARIMAX model with all variables
arimax_all <- Arima(train_interest, order = c(1,1,2), xreg = train_xreg)
print("ARIMAX(1,1,2) with all variables:")
summary(arimax_all)

# Check residuals of the full model
checkresiduals(arimax_all)

# Try individual predictors to see their individual impact
# Exchange Rate only
arimax_er <- Arima(train_interest, order = c(1,1,2), 
                   xreg = as.matrix(train_xreg[,"ExchangeRate"]))
print("ARIMAX with Exchange Rate only:")
summary(arimax_er)

# VIX only
arimax_vix <- Arima(train_interest, order = c(1,1,2), 
                    xreg = as.matrix(train_xreg[,"VIX"]))
print("ARIMAX with VIX only:")
summary(arimax_vix)

# TEDSpread only
arimax_ted <- Arima(train_interest, order = c(1,1,2), 
                    xreg = as.matrix(train_xreg[,"TEDSpread"]))
print("ARIMAX with TEDSpread only:")
summary(arimax_ted)

# EFFR only
arimax_effr <- Arima(train_interest, order = c(1,1,2), 
                     xreg = as.matrix(train_xreg[,"EFFR"]))
print("ARIMAX with EFFR only:")
summary(arimax_effr)

# Try pairs of variables
# Exchange Rate + VIX
arimax_er_vix <- Arima(train_interest, order = c(1,1,2), 
                       xreg = train_xreg[,c("ExchangeRate","VIX")])
print("ARIMAX with Exchange Rate + VIX:")
summary(arimax_er_vix)

# Exchange Rate + TEDSpread
arimax_er_ted <- Arima(train_interest, order = c(1,1,2), 
                       xreg = train_xreg[,c("ExchangeRate","TEDSpread")])
print("ARIMAX with Exchange Rate + TEDSpread:")
summary(arimax_er_ted)

# Exchange Rate + EFFR
arimax_er_effr <- Arima(train_interest, order = c(1,1,2), 
                        xreg = train_xreg[,c("ExchangeRate","EFFR")])
print("ARIMAX with Exchange Rate + EFFR:")
summary(arimax_er_effr)

# VIX + TEDSpread
arimax_vix_ted <- Arima(train_interest, order = c(1,1,2), 
                        xreg = train_xreg[,c("VIX","TEDSpread")])
print("ARIMAX with VIX + TEDSpread:")
summary(arimax_vix_ted)

# VIX + EFFR
arimax_vix_effr <- Arima(train_interest, order = c(1,1,2), 
                         xreg = train_xreg[,c("VIX","EFFR")])
print("ARIMAX with VIX + EFFR:")
summary(arimax_vix_effr)

# TEDSpread + EFFR
arimax_ted_effr <- Arima(train_interest, order = c(1,1,2), 
                         xreg = train_xreg[,c("TEDSpread","EFFR")])
print("ARIMAX with TEDSpread + EFFR:")
summary(arimax_ted_effr)

# Try triplets of variables
# Exchange Rate + VIX + TEDSpread
arimax_er_vix_ted <- Arima(train_interest, order = c(1,1,2), 
                           xreg = train_xreg[,c("ExchangeRate","VIX","TEDSpread")])
print("ARIMAX with Exchange Rate + VIX + TEDSpread:")
summary(arimax_er_vix_ted)

# Exchange Rate + VIX + EFFR
arimax_er_vix_effr <- Arima(train_interest, order = c(1,1,2), 
                            xreg = train_xreg[,c("ExchangeRate","VIX","EFFR")])
print("ARIMAX with Exchange Rate + VIX + EFFR:")
summary(arimax_er_vix_effr)

# Exchange Rate + TEDSpread + EFFR
arimax_er_ted_effr <- Arima(train_interest, order = c(1,1,2), 
                            xreg = train_xreg[,c("ExchangeRate","TEDSpread","EFFR")])
print("ARIMAX with Exchange Rate + TEDSpread + EFFR:")
summary(arimax_er_ted_effr)

# VIX + TEDSpread + EFFR
arimax_vix_ted_effr <- Arima(train_interest, order = c(1,1,2), 
                             xreg = train_xreg[,c("VIX","TEDSpread","EFFR")])
print("ARIMAX with VIX + TEDSpread + EFFR:")
summary(arimax_vix_ted_effr)

# Compare AICc values to identify best predictor combination
arimax_models <- list(
  "All Variables" = arimax_all,
  "ExchangeRate" = arimax_er,
  "VIX" = arimax_vix,
  "TEDSpread" = arimax_ted,
  "EFFR" = arimax_effr,
  "ExchangeRate+VIX" = arimax_er_vix,
  "ExchangeRate+TEDSpread" = arimax_er_ted,
  "ExchangeRate+EFFR" = arimax_er_effr,
  "VIX+TEDSpread" = arimax_vix_ted,
  "VIX+EFFR" = arimax_vix_effr,
  "TEDSpread+EFFR" = arimax_ted_effr,
  "ExchangeRate+VIX+TEDSpread" = arimax_er_vix_ted,
  "ExchangeRate+VIX+EFFR" = arimax_er_vix_effr,
  "ExchangeRate+TEDSpread+EFFR" = arimax_er_ted_effr,
  "VIX+TEDSpread+EFFR" = arimax_vix_ted_effr
)

# Extract AICc values
aicc_values <- sapply(arimax_models, function(model) model$aicc)

# Create a data frame for better display
model_comparison <- data.frame(
  Predictors = names(arimax_models),
  AICc = aicc_values
)

# Sort by AICc (lower is better)
model_comparison <- model_comparison[order(model_comparison$AICc), ]
print("Predictor Combinations Ranked by AICc (lower is better):")
print(model_comparison)

# Identify the best predictor combination
best_predictors <- model_comparison$Predictors[1]
print(paste("Best predictor combination based on AICc:", best_predictors))

# Extract the best model (TEDSpread+EFFR)
best_arimax <- arimax_ted_effr  # This should be the model with TEDSpread and EFFR

# Prepare test data for forecasting with only the best predictors
test_xreg_best <- test_xreg[, c("TEDSpread", "EFFR")]

# Generate forecasts
arimax_forecast <- forecast(best_arimax, xreg = test_xreg_best, h = length(test_interest))

# Plot the forecast
plot(arimax_forecast, main = "ARIMAX(1,1,2) with TEDSpread+EFFR: Forecast vs Actual",
     xlab = "Time", ylab = "Change in Interest Rate")
lines(test_interest, col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1)

# Check the accuracy of the ARIMAX forecast
print("ARIMAX(1,1,2) with TEDSpread+EFFR Forecast Accuracy:")
arimax_accuracy <- accuracy(arimax_forecast, test_interest)
print(arimax_accuracy)

# Display model details
print("ARIMAX(1,1,2) with TEDSpread+EFFR Model Details:")
summary(best_arimax)

# Check residuals
checkresiduals(best_arimax)

# Model 2: VIX+EFFR (Second best AICc)
# Prepare test data for forecasting
test_xreg_vix_effr <- test_xreg[, c("VIX", "EFFR")]

# Generate forecasts
arimax_vix_effr_forecast <- forecast(arimax_vix_effr, xreg = test_xreg_vix_effr, h = length(test_interest))

# Plot the forecast
plot(arimax_vix_effr_forecast, main = "ARIMAX(1,1,2) with VIX+EFFR: Forecast vs Actual",
     xlab = "Time", ylab = "Change in Interest Rate")
lines(test_interest, col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1)

# Check the accuracy
print("ARIMAX(1,1,2) with VIX+EFFR Forecast Accuracy:")
arimax_vix_effr_accuracy <- accuracy(arimax_vix_effr_forecast, test_interest)
print(arimax_vix_effr_accuracy)

# Model 3: ExchangeRate+VIX+EFFR (Third best AICc)
# Prepare test data for forecasting
test_xreg_er_vix_effr <- test_xreg[, c("ExchangeRate", "VIX", "EFFR")]

# Generate forecasts
arimax_er_vix_effr_forecast <- forecast(arimax_er_vix_effr, xreg = test_xreg_er_vix_effr, h = length(test_interest))

# Plot the forecast
plot(arimax_er_vix_effr_forecast, main = "ARIMAX(1,1,2) with ExchangeRate+VIX+EFFR: Forecast vs Actual",
     xlab = "Time", ylab = "Change in Interest Rate")
lines(test_interest, col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1)

# Check the accuracy
print("ARIMAX(1,1,2) with ExchangeRate+VIX+EFFR Forecast Accuracy:")
arimax_er_vix_effr_accuracy <- accuracy(arimax_er_vix_effr_forecast, test_interest)
print(arimax_er_vix_effr_accuracy)

# Create a data frame to compare accuracy metrics
accuracy_comparison <- data.frame(
  Model = c("TEDSpread+EFFR", "VIX+EFFR", "ExchangeRate+VIX+EFFR"),
  RMSE = c(
    arimax_accuracy["Test set", "RMSE"],
    arimax_vix_effr_accuracy["Test set", "RMSE"],
    arimax_er_vix_effr_accuracy["Test set", "RMSE"]
  ),
  MAE = c(
    arimax_accuracy["Test set", "MAE"],
    arimax_vix_effr_accuracy["Test set", "MAE"],
    arimax_er_vix_effr_accuracy["Test set", "MAE"]
  ),
  MAPE = c(
    arimax_accuracy["Test set", "MAPE"],
    arimax_vix_effr_accuracy["Test set", "MAPE"],
    arimax_er_vix_effr_accuracy["Test set", "MAPE"]
  ),
  MASE = c(
    arimax_accuracy["Test set", "MASE"],
    arimax_vix_effr_accuracy["Test set", "MASE"],
    arimax_er_vix_effr_accuracy["Test set", "MASE"]
  )
)

# Sort by RMSE (lower is better)
accuracy_comparison <- accuracy_comparison[order(accuracy_comparison$RMSE), ]

# Print the comparison table
print("ARIMAX Models Accuracy Comparison (sorted by RMSE, lower is better):")
print(accuracy_comparison)

# Identify the best model
best_model <- accuracy_comparison$Model[1]
print(paste("Best ARIMAX model based on RMSE:", best_model))

# Calculate percentage improvement over worst model
worst_rmse <- max(accuracy_comparison$RMSE)
best_rmse <- min(accuracy_comparison$RMSE)
improvement <- ((worst_rmse - best_rmse) / worst_rmse) * 100
print(paste("The best model shows a", round(improvement, 2), "% improvement in RMSE over the worst model"))

# 6.4 
# VECTOR AUTOREGRESSION (VAR) MODEL

# Examine data with plots
autoplot(ts(nasdaq_monthly$InterestRate, frequency = 12), 
         main = "Interest Rate Time Series")

# Create multivariate time series with interest rates and predictors
var_data <- cbind(
  InterestRate = nasdaq_monthly$InterestRate,
  ExchangeRate = nasdaq_monthly$ExchangeRate,
  VIX = nasdaq_monthly$VIX,
  TEDSpread = nasdaq_monthly$TEDSpread,
  EFFR = nasdaq_monthly$EFFR
)

# Convert to time series matrix
var_ts <- ts(var_data, frequency = 12, 
             start = c(as.numeric(format(min(nasdaq_monthly$Date), "%Y")), 
                       as.numeric(format(min(nasdaq_monthly$Date), "%m"))))

# Plot all variables
autoplot(var_ts, facets = TRUE)

# Check for stationarity of all variables using ADF tests
print("Stationarity tests for Interest Rate:")
ur.df(var_ts[, "InterestRate"], type = "drift", lags = 1) %>% summary()

print("After differencing:")
ur.df(diff(var_ts[, "InterestRate"]), type = "drift", lags = 1) %>% summary()

# Check other variables (just examining a couple for brevity)
print("Stationarity tests for EFFR:")
ur.df(var_ts[, "EFFR"], type = "drift", lags = 1) %>% summary()

print("Stationarity tests for VIX:")
ur.df(var_ts[, "VIX"], type = "drift", lags = 1) %>% summary()

# Difference all variables for consistency (based on our earlier findings)
var_diff <- diff(var_ts)

# View differenced series
autoplot(var_diff, facets = TRUE)

# Split into training and testing sets
train_var_diff <- window(var_diff, end = time(var_diff)[n_train-1])
test_var_diff <- window(var_diff, start = time(var_diff)[n_train])

# Select optimal lag order
lag_selection <- VARselect(train_var_diff, lag.max = 10, type = "const")
print("Optimal lag selection:")
print(lag_selection$selection)

# Try different VAR models with increasing lag orders
var_mod1 <- VAR(train_var_diff, p = 1, type = "const")
summary(var_mod1)

var_mod2 <- VAR(train_var_diff, p = 2, type = "const")
summary(var_mod2)

var_mod3 <- VAR(train_var_diff, p = 3, type = "const")
summary(var_mod3)

# Check for serial correlation in residuals for first model
serial_test1 <- serial.test(var_mod1, lags.pt = 10, type = "PT.asymptotic")
print("Serial Correlation Test for VAR(1):")
print(serial_test1)

# Check for serial correlation in residuals for second model
serial_test2 <- serial.test(var_mod2, lags.pt = 10, type = "PT.asymptotic")
print("Serial Correlation Test for VAR(2):")
print(serial_test2)

# Check for serial correlation in residuals for third model
serial_test3 <- serial.test(var_mod3, lags.pt = 10, type = "PT.asymptotic")
print("Serial Correlation Test for VAR(3):")
print(serial_test3)

# VAR(2) captures the important dynamics between interest rates and economic 
# indicators with sufficient lag structure to model the relationships adequately, 
# without introducing unnecessary complexity. This makes VAR(2) the optimal choice
# among the three models for your interest rate forecasting project.

# Select best model based on results (assuming model 2 for this example)
best_var <- var_mod2  

# Generate forecasts
var_forecast <- predict(best_var, n.ahead = length(test_var_diff[, 1]))

# Extract interest rate forecasts
interest_forecast <- var_forecast$fcst$InterestRate[, 1]

# Plot the forecast vs actual
plot(var_forecast, names = "InterestRate")
lines(test_var_diff[, "InterestRate"], col = "red")
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("black", "red"), lty = 1)

# Calculate forecast accuracy (we need to match format for accuracy function)
# Convert forecast to ts object with appropriate time properties
forecast_ts <- ts(interest_forecast, 
                  start = time(test_var_diff)[1], 
                  frequency = 12)

# Calculate accuracy metrics
var_accuracy <- accuracy(forecast_ts, test_var_diff[, "InterestRate"])
print("VAR Model Forecast Accuracy:")
print(var_accuracy)

# 6.5 check residuals

# 1. ETS benchmark
checkresiduals(ets_model, main = "ETS(M,N,N) Residual Diagnostics")

# 2. ARIMA(1,1,2)
checkresiduals(arima_112, main = "ARIMA(1,1,2) Residual Diagnostics")

# 3. ARIMAX (TEDSpread + EFFR)
checkresiduals(best_arimax, main = "ARIMAX(TEDSpread+EFFR) Residual Diagnostics")

# 4. VAR(2)
#   a) Residual time series + ACF + Q–Q
resid_var2 <- residuals(best_var)
ts.plot(resid_var2[, "InterestRate"], main = "VAR(2) InterestRate Residuals", ylab = "Residual")
acf(resid_var2[, "InterestRate"], main = "VAR(2) Residual ACF")
qqnorm(resid_var2[, "InterestRate"], main = "VAR(2) Residual Q–Q Plot"); qqline(resid_var2[, "InterestRate"])

#   b) Ljung–Box (Portmanteau) test up to lag 10 for the VAR residuals
serial.test(best_var, lags.pt = 10, type = "PT.asymptotic")


# PART 7: MODEL COMPARISON AND EVALUATION

# 1. Back-transform all forecasts to original scale
# Get the last value of the original interest rate series before differencing
last_train_value <- as.numeric(tail(train_original, 1))  # Convert to numeric

# ETS model (already in original scale)
# No need to back-transform

# ARIMA model - back-transform by adding cumulative sum to last value
arima_forecasts_original <- last_train_value + cumsum(as.numeric(best_arima_forecast$mean))
arima_forecasts_ts <- ts(arima_forecasts_original, 
                         start = time(test_original)[1], 
                         frequency = 12)

# ARIMAX model with best predictors (TEDSpread+EFFR) - back-transform
arimax_forecasts_original <- last_train_value + cumsum(as.numeric(arimax_forecast$mean))
arimax_forecasts_ts <- ts(arimax_forecasts_original, 
                          start = time(test_original)[1], 
                          frequency = 12)

# VAR(2) model - back-transform
var_forecasts_original <- last_train_value + cumsum(as.numeric(interest_forecast))
var_forecasts_ts <- ts(var_forecasts_original, 
                       start = time(test_original)[1], 
                       frequency = 12)

# 2. Calculate accuracy metrics for all models on original scale
ets_accuracy_orig <- accuracy(ets_forecast, test_original)
arima_accuracy_orig <- accuracy(arima_forecasts_ts, test_original)
arimax_accuracy_orig <- accuracy(arimax_forecasts_ts, test_original)
var_accuracy_orig <- accuracy(var_forecasts_ts, test_original)

# 3. Create comparison table
comparison_table <- data.frame(
  Model = c("ETS", best_model_name, "ARIMAX(TEDSpread+EFFR)", "VAR(2)"),
  RMSE = c(ets_accuracy_orig["Test set", "RMSE"],
           arima_accuracy_orig["Test set", "RMSE"],
           arimax_accuracy_orig["Test set", "RMSE"],
           var_accuracy_orig["Test set", "RMSE"]),
  MAE = c(ets_accuracy_orig["Test set", "MAE"],
          arima_accuracy_orig["Test set", "MAE"],
          arimax_accuracy_orig["Test set", "MAE"],
          var_accuracy_orig["Test set", "MAE"])
)

# Sort by RMSE (lower is better)
comparison_table <- comparison_table[order(comparison_table$RMSE), ]

# Print comparison
print("Model Comparison on Original Scale (lower values are better):")
print(comparison_table)

# 4. Create visualization using a loop for efficiency
# Set up plotting parameters
models <- list(
  ARIMA = list(
    values = arima_forecasts_original,
    ts_obj = arima_forecasts_ts,
    color = "blue",
    shade = "lightblue",
    se = sd(as.numeric(best_arima_forecast$residuals)),
    title = best_model_name
  ),
  ARIMAX = list(
    values = arimax_forecasts_original,
    ts_obj = arimax_forecasts_ts,
    color = "green",
    shade = "lightgreen",
    se = sd(as.numeric(arimax_forecast$residuals)),
    title = "ARIMAX(TEDSpread+EFFR)"
  ),
  VAR = list(
    values = var_forecasts_original,
    ts_obj = var_forecasts_ts,
    color = "purple",
    shade = "lavender",
    se = sd(as.numeric(residuals(var_mod2)[,"InterestRate"])),
    title = "VAR(2)"
  ),
  ETS = list(
    values = as.numeric(ets_forecast$mean),
    ts_obj = ets_forecast$mean,
    color = "orange",
    shade = "lightyellow",
    se = sd(as.numeric(ets_forecast$residuals)),
    title = "ETS"
  )
)

# Create a plot window first
plot(window(interest_ts, start = time(interest_ts)[n_train - 12]), 
     xlim = c(time(interest_ts)[n_train - 12], time(test_original)[length(test_original)]),
     main = "Comparison of Interest Rate Forecast Models",
     xlab = "Time", ylab = "Interest Rate")

# Add vertical line at start of forecast period
abline(v = time(test_original)[1], lty = 2, col = "gray")

# Add each model's forecast with different colors
for (model_name in names(models)) {
  model <- models[[model_name]]
  lines(model$ts_obj, col = model$color, lwd = 1.5)
}

# Add actual values
lines(test_original, col = "red", lwd = 2)

# Add legend
legend("topleft", 
       legend = c("Historical", names(models), "Actual"),
       col = c("black", sapply(models, function(m) m$color), "red"),
       lty = 1, lwd = c(1, rep(1.5, length(models)), 2))

# 5. Individual plots with confidence intervals
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Loop through models to create individual plots
for (model_name in names(models)) {
  model <- models[[model_name]]
  
  # Set up the plot
  plot(window(interest_ts, start = time(interest_ts)[n_train - 6]), 
       xlim = c(time(interest_ts)[n_train - 6], time(test_original)[length(test_original)]),
       main = paste(model$title, "Forecast"),
       xlab = "Time", ylab = "Interest Rate")
  
  # Add vertical line at forecast start
  abline(v = time(test_original)[1], lty = 2, col = "gray")
  
  # Add forecast
  lines(model$ts_obj, col = model$color, lwd = 1.5)
  
  # Add actual values
  lines(test_original, col = "red", lwd = 2)
  
  # Add confidence intervals if not ETS (ETS has built-in CIs)
  if (model_name != "ETS") {
    # Create time points for polygon
    time_points <- time(model$ts_obj)
    
    # Create upper and lower bounds
    upper <- model$values + 1.96 * model$se
    lower <- model$values - 1.96 * model$se
    
    # Create time series objects for bounds
    upper_ts <- ts(upper, start = time(model$ts_obj)[1], frequency = 12)
    lower_ts <- ts(lower, start = time(model$ts_obj)[1], frequency = 12)
    
    # Add shaded region
    polygon(c(time_points, rev(time_points)),
            c(upper, rev(lower)),
            col = adjustcolor(model$shade, alpha.f = 0.3), 
            border = NA)
  } else {
    # For ETS, just use the plot function which already includes CIs
    plot(ets_forecast, main = "ETS Forecast", xlab = "Time", ylab = "Interest Rate")
    lines(test_original, col = "red", lwd = 2)
  }
  
  # Add legend
  legend("topleft", 
         legend = c("Forecast", "Actual", "95% CI"),
         col = c(model$color, "red", model$shade),
         lty = c(1, 1, NA),
         fill = c(NA, NA, model$shade),
         border = c(NA, NA, NA))
}

# Reset plot layout
par(mfrow = c(1, 1))

# 6. Extract the best model based on RMSE
best_overall_model <- comparison_table$Model[1]
print(paste("The best model for forecasting interest rates is:", best_overall_model))

# PART 8: FORECASTING WITH VAR(2) 

# 1. Generate forecasts with our best model (VAR(2))
print("Generating forecasts with VAR(2) model...")
var_forecast <- predict(best_var, n.ahead = 12)  # Forecast 12 months ahead

# 2. Extract interest rate forecasts
interest_forecast_future <- var_forecast$fcst$InterestRate[, 1]
interest_forecast_lower <- var_forecast$fcst$InterestRate[, 2]  # Lower prediction interval
interest_forecast_upper <- var_forecast$fcst$InterestRate[, 3]  # Upper prediction interval

# 3. Back-transform to original scale
# Get the last observed value
last_observed_value <- tail(interest_ts, 1)

# Create future dates for forecasting
forecast_dates <- seq(from = max(nasdaq_monthly$Date) + 30, by = "month", length.out = 12)
forecast_years <- as.numeric(format(forecast_dates, "%Y"))
forecast_months <- as.numeric(format(forecast_dates, "%m"))
forecast_start <- c(forecast_years[1], forecast_months[1])

# Back-transform the differenced forecasts to original scale
forecast_original <- numeric(12)
forecast_original[1] <- last_observed_value + interest_forecast_future[1]
for (i in 2:12) {
  forecast_original[i] <- forecast_original[i-1] + interest_forecast_future[i]
}

# Create time series object for plotting
forecast_ts <- ts(forecast_original, 
                  start = forecast_start, 
                  frequency = 12)

# 4. Create confidence intervals
forecast_lower <- numeric(12)
forecast_lower[1] <- last_observed_value + interest_forecast_lower[1]
forecast_upper <- numeric(12)
forecast_upper[1] <- last_observed_value + interest_forecast_upper[1]

for (i in 2:12) {
  forecast_lower[i] <- forecast_lower[i-1] + interest_forecast_lower[i]
  forecast_upper[i] <- forecast_upper[i-1] + interest_forecast_upper[i]
}

forecast_lower_ts <- ts(forecast_lower, start = forecast_start, frequency = 12)
forecast_upper_ts <- ts(forecast_upper, start = forecast_start, frequency = 12)

# 5. Plot the forecasts
# Start with historical data
plot(interest_ts, 
     xlim = c(time(interest_ts)[1], time(forecast_ts)[12]),
     ylim = range(c(interest_ts, forecast_upper_ts, forecast_lower_ts)),
     main = "Interest Rate Forecast with VAR(2) Model",
     xlab = "Year", ylab = "Interest Rate")

# Add forecasts
lines(forecast_ts, col = "blue", lwd = 2)

# Add shaded confidence intervals
polygon(c(time(forecast_ts), rev(time(forecast_ts))),
        c(forecast_lower, rev(forecast_upper)),
        col = adjustcolor("lightblue", alpha.f = 0.3), 
        border = NA)

# Add legend
legend("topleft", 
       legend = c("Historical", "Forecast", "95% CI"),
       col = c("black", "blue", "lightblue"),
       lty = c(1, 1, NA),
       pch = c(NA, NA, 15),
       pt.cex = 2)

# 6. Print forecast values
forecast_table <- data.frame(
  Date = format(forecast_dates, "%b %Y"),
  Forecast = round(forecast_original, 2),
  Lower_95 = round(forecast_lower, 2),
  Upper_95 = round(forecast_upper, 2)
)

print("12-Month Interest Rate Forecast:")
print(forecast_table)

