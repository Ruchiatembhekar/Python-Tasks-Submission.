import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample house dataset
def create_sample_dataset():
    """Create a sample house price dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    rooms = np.random.randint(1, 6, n_samples)
    size = np.random.randint(500, 3000, n_samples)  # sq ft
    location_score = np.random.randint(1, 11, n_samples)  # 1-10 rating
    age = np.random.randint(0, 50, n_samples)  # years old
    
    # Generate price based on features with some noise
    price = (
        rooms * 15000 +  # $15k per room
        size * 100 +     # $100 per sq ft  
        location_score * 5000 -  # $5k per location point
        age * 500 +      # depreciation
        np.random.normal(0, 10000, n_samples)  # noise
    )
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'rooms': rooms,
        'size_sqft': size,
        'location_score': location_score,
        'age_years': age,
        'price': price
    })
    
    return data

def analyze_dataset(df):
    """Perform basic data analysis"""
    print("🏠 HOUSE PRICE DATASET ANALYSIS")
    print("=" * 40)
    
    print("\n📊 Dataset Overview:")
    print(f"Total Houses: {len(df)}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"\n📈 Dataset Statistics:")
    print(df.describe())
    
    # Correlation analysis
    print("\n🔗 Feature Correlations with Price:")
    correlations = df.corr()['price'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'price':
            print(f"{feature}: {corr:.3f}")
    
    return df

def create_visualizations(df):
    """Create data visualizations"""
    plt.figure(figsize=(15, 12))
    
    # Price distribution
    plt.subplot(2, 3, 1)
    plt.hist(df['price'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('House Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    
    # Rooms vs Price
    plt.subplot(2, 3, 2)
    plt.scatter(df['rooms'], df['price'], alpha=0.6, color='blue')
    plt.title('Rooms vs Price')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price ($)')
    
    # Size vs Price
    plt.subplot(2, 3, 3)
    plt.scatter(df['size_sqft'], df['price'], alpha=0.6, color='green')
    plt.title('Size vs Price')
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price ($)')
    
    # Location vs Price
    plt.subplot(2, 3, 4)
    plt.scatter(df['location_score'], df['price'], alpha=0.6, color='red')
    plt.title('Location Score vs Price')
    plt.xlabel('Location Score (1-10)')
    plt.ylabel('Price ($)')
    
    # Age vs Price
    plt.subplot(2, 3, 5)
    plt.scatter(df['age_years'], df['price'], alpha=0.6, color='orange')
    plt.title('Age vs Price')
    plt.xlabel('Age (years)')
    plt.ylabel('Price ($)')
    
    # Correlation heatmap
    plt.subplot(2, 3, 6)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_model(df):
    """Train linear regression model"""
    print("\n🤖 TRAINING LINEAR REGRESSION MODEL")
    print("=" * 40)
    
    # Prepare features and target
    features = ['rooms', 'size_sqft', 'location_score', 'age_years']
    X = df[features]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n📊 Model Performance:")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Test RMSE: ${test_rmse:,.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Feature importance
    print(f"\n🎯 Feature Importance (Coefficients):")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: ${coef:,.2f}")
    print(f"Intercept: ${model.intercept_:,.2f}")
    
    return model, X_test, y_test, y_pred_test

def make_predictions(model):
    """Interactive prediction tool"""
    print("\n🏡 MAKE PRICE PREDICTIONS")
    print("=" * 30)
    
    while True:
        try:
            print("\nEnter house details:")
            rooms = int(input("Number of rooms (1-5): "))
            size = int(input("Size in sq ft (500-3000): "))
            location = int(input("Location score (1-10): "))
            age = int(input("Age in years (0-50): "))
            
            # Make prediction
            features = np.array([[rooms, size, location, age]])
            predicted_price = model.predict(features)[0]
            
            print(f"\n🏠 Predicted House Price: ${predicted_price:,.2f}")
            
            another = input("\nPredict another house? (y/n): ").lower()
            if another != 'y':
                break
                
        except ValueError:
            print("Please enter valid numbers!")
        except KeyboardInterrupt:
            break

def main():
    """Main execution function"""
    print("🏠 HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 50)
    
    # Create and load dataset
    print("\n📁 Creating sample dataset...")
    df = create_sample_dataset()
    df.to_csv('house_dataset.csv', index=False)
    print("Dataset saved as 'house_dataset.csv'")
    
    # Analyze dataset
    df = analyze_dataset(df)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(df)
    
    # Train model
    model, X_test, y_test, y_pred = train_model(df)
    
    # Interactive predictions
    make_predictions(model)
    
    print("\n✅ House Price Prediction Complete!")
    print("Check 'house_price_analysis.png' for visualizations")

if __name__ == "__main__":
    main()