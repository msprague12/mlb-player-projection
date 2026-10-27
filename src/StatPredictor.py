import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaseball import batting_stats
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

#Target stats to predict
target_stats = ['AVG', 'OBP', 'SLG', 'HR', 'RBI', 'wOBA', 'wRC+']

#Features for prediction
feature_names = ['Age', 'G', 'PA', 'AVG', 'OBP', 'SLG', 'HR', 'RBI', 'BB%', 'K%', 
                 'ISO', 'BABIP', 'wOBA', 'wRC+', 'WAR']

#Get training data from existing databases using pybaseball api package 
def get_training_data(start_year=2013, end_year=2023, min_pa=100):
    print("Collecting training data...")
    X_list = []
    y_list = []
    
    #Iterate through consecutive seasons
    for year in range(start_year, end_year):
        print(f"Processing {year}-{year+1} seasons...")
        
        #Get current and next season data
        current_season = batting_stats(year, year, qual=0)
        next_season = batting_stats(year+1, year+1, qual=0)
        
        #Filter to minimum plate appearances
        current_season = current_season[current_season['PA'] >= min_pa]
        next_season = next_season[next_season['PA'] >= min_pa]
        
        #Find players who appear in both seasons
        common_players = set(current_season['IDfg']).intersection(set(next_season['IDfg']))
        
        for player_id in common_players:
            current_stats = current_season[current_season['IDfg'] == player_id]
            next_stats = next_season[next_season['IDfg'] == player_id]
            
            if len(current_stats) == 1 and len(next_stats) == 1:
                #Extract features and targets
                X_row = current_stats[feature_names].values.flatten()
                y_row = next_stats[target_stats].values.flatten()
                
                X_list.append(X_row)
                y_list.append(y_row)
    
    return np.array(X_list), np.array(y_list)

#######Train models for each target statistic########
def train_models(X, y, save=True):
    print("Training projection models...")
    models = {}
    
    for i, stat in enumerate(target_stats):
        print(f"Training model for {stat}...")
        
        #Extract target variable
        y_target = y[:, i]
        
        #Train random forrest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y_target)
        y_rf_pred = rf_model.predict(X)
        rf_mse = mean_squared_error(y_target, y_rf_pred)
        
        #Train xgboost model 
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X, y_target)
        y_xgb_pred = xgb_model.predict(X)
        xgb_mse = mean_squared_error(y_target, y_xgb_pred)

        #Print mean squared error for both models 
        print(f"{stat} - RF MSE: {rf_mse:.4f}, XGB MSE: {xgb_mse:.4f}")

        
        models[stat] = {
            'rf': rf_model,
            'xgb': xgb_model
        }
    
    if save:
        with open("projection_models.pkl", 'wb') as f:
            pickle.dump(models, f)
        print("Models saved to projection_models.pkl")
    
    return models

#Load existing models or train new ones
def load_or_train_models():
    try:
        with open("projection_models.pkl", 'rb') as f:
            models = pickle.load(f)
        print("Loaded existing models")
        return models
    except:
        print("No existing models found. Training new models...")
        X, y = get_training_data()
        return train_models(X, y)

#Find a player in a given season 
def find_player(player_name, year):
    try:
        #Get data for the specified season
        season_data = batting_stats(year, year, qual=0)
        
        #Filter by minimum plate appearances
        season_data = season_data[season_data['PA'] >= 100]
        
        #Find players matching the name
        matches = season_data[season_data['Name'].str.contains(player_name, case=False)]
        
        if len(matches) == 0:
            print(f"No players found matching '{player_name}' in {year}")
            return None
        elif len(matches) > 1:
            print(f"Multiple players found matching '{player_name}' in {year}:")
            print(matches[['Name', 'Team', 'G', 'PA']].to_string(index=False))
            player_idx = int(input("Enter the index of the correct player: "))
            return matches.iloc[player_idx]
        else:
            return matches.iloc[0]
    except Exception as e:
        print(f"Error finding player: {e}")
        return None

#Project player statistics for future seasons
def project_player(player_name, current_year, projection_years, models):
    #Find the player
    player_data = find_player(player_name, current_year)
    if player_data is None:
        return None
    
    #Extract player info
    player_age = player_data['Age']
    player_name_full = player_data['Name']
    
    #Create feature vector
    X = player_data[feature_names].values.reshape(1, -1)
    
    #Create projections dataframe
    projections = pd.DataFrame(columns=['Year', 'Age', 'ProjectionType'] + target_stats)
    
    #Add current year data
    current_row = {
        'Year': current_year,
        'Age': player_age,
        'ProjectionType': 'Actual'
    }
    for stat in target_stats:
        current_row[stat] = player_data[stat]
    
    projections = pd.concat([projections, pd.DataFrame([current_row])], ignore_index=True)
    
    #Generate projections for future years
    for year_offset in range(1, projection_years + 1):
        future_year = current_year + year_offset
        future_age = player_age + year_offset
        
        #Use a copy of the features for projection
        X_future = X.copy()
        
        #Update age for future year
        age_index = feature_names.index('Age')
        X_future[0, age_index] = future_age
        
        #Create prediction rows
        rf_row = {
            'Year': future_year,
            'Age': future_age,
            'ProjectionType': 'RandomForest'
        }
        
        xgb_row = {
            'Year': future_year,
            'Age': future_age,
            'ProjectionType': 'XGBoost'
        }
        
        #Generate projections for each stat
        for stat in target_stats:
            rf_model = models[stat]['rf']
            xgb_model = models[stat]['xgb']
            
            rf_pred = rf_model.predict(X_future)[0]
            xgb_pred = xgb_model.predict(X_future)[0]
            
            rf_row[stat] = rf_pred
            xgb_row[stat] = xgb_pred
            
            #Update feature array with predicted values for cascading projections
            if stat in feature_names:
                feat_index = feature_names.index(stat)
                #Use average of models for next projection
                X_future[0, feat_index] = (rf_pred + xgb_pred) / 2
        
        projections = pd.concat([projections, pd.DataFrame([rf_row, xgb_row])], ignore_index=True)
    
    return projections, player_name_full

#Visualize a player's projected statistic
def visualize_projections(projections, player_name, stat):
    plt.figure(figsize=(10, 6))
    
    #Plot actual values
    actual = projections[projections['ProjectionType'] == 'Actual']
    plt.scatter(actual['Year'], actual[stat], color='black', s=100, label='Actual')
    
    #Plot RF projections
    rf_proj = projections[projections['ProjectionType'] == 'RandomForest']
    plt.plot(rf_proj['Year'], rf_proj[stat], 'b-o', label='Random Forest')
    
    #Plot XGB projections
    xgb_proj = projections[projections['ProjectionType'] == 'XGBoost']
    plt.plot(xgb_proj['Year'], xgb_proj[stat], 'r-o', label='XGBoost')
    
    plt.title(f"{player_name} - {stat} Projection")
    plt.xlabel('Year')
    plt.ylabel(stat)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def main():
    #Load or train models
    models = load_or_train_models()
    
    #Prompt user to enter player name
    while True:
        print("\nEnter player details (or 'quit' to exit)")
        player_name = input("Player name: ")
        
        #Allow user to exit 
        if player_name.lower() == 'quit':
            break
        
        try:
            current_year = int(input("Starting year: "))
            projection_years = int(input("Number of years to project: "))
            
            #Generate projections
            result = project_player(player_name, current_year, projection_years, models)
            
            if result is not None:
                projections, full_name = result
                
                #Display projections
                print(f"\nProjections for {full_name}:")
                print(projections.to_string(index=False))
                
                #Ask user which stats to visualize
                print("\nVisualize projections:")
                for i, stat in enumerate(target_stats):
                    print(f"{i+1}. {stat}")
                
                while True:
                    choice = input("\nEnter stat number to visualize (or 'done'): ")
                    if choice.lower() == 'done':
                        break
                    
                    try:
                        stat_idx = int(choice) - 1
                        if 0 <= stat_idx < len(target_stats):
                            stat = target_stats[stat_idx]
                            plt = visualize_projections(projections, full_name, stat)
                            plt.show()
                        else:
                            print("Invalid selection")
                    except:
                        print("Please enter a valid number")
        
        except ValueError:
            print("Please enter valid numbers")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()