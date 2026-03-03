"""
Dish Dataset Module for the Food Waste Prediction System.

Handles loading, generating, and preprocessing the dish/recipe dataset.
Includes multi-hot encoding of ingredients and dish feature computation.
"""

import os
import sys
import pandas as pd
import numpy as np
from ast import literal_eval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Definitions
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DISHES = [
    # Veg Indian
    {"dish_name": "Aloo Gobi", "ingredients": "potato,cauliflower,onion,tomato,oil,turmeric,cumin,coriander", "quantities_g": "200,200,100,80,20,3,5,5", "cuisine": "indian", "menu_type": "veg", "calories": 210, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Palak Paneer", "ingredients": "spinach,paneer,onion,tomato,cream,garlic,ginger,oil,turmeric", "quantities_g": "300,200,100,80,30,10,10,20,3", "cuisine": "indian", "menu_type": "veg", "calories": 280, "prep_time_min": 35, "servings": 4},
    {"dish_name": "Dal Tadka", "ingredients": "lentils,onion,tomato,garlic,ginger,oil,turmeric,cumin,coriander", "quantities_g": "200,100,80,10,10,20,3,5,5", "cuisine": "indian", "menu_type": "veg", "calories": 230, "prep_time_min": 40, "servings": 6},
    {"dish_name": "Vegetable Biryani", "ingredients": "rice,potato,carrot,peas,onion,tomato,oil,yogurt,garam_masala", "quantities_g": "300,100,80,60,120,80,30,60,10", "cuisine": "indian", "menu_type": "veg", "calories": 350, "prep_time_min": 60, "servings": 6},
    {"dish_name": "Chana Masala", "ingredients": "chickpea,onion,tomato,garlic,ginger,oil,turmeric,cumin,coriander", "quantities_g": "250,120,100,15,15,25,3,5,5", "cuisine": "indian", "menu_type": "veg", "calories": 260, "prep_time_min": 45, "servings": 4},
    {"dish_name": "Paneer Butter Masala", "ingredients": "paneer,butter,cream,onion,tomato,garlic,ginger,oil,garam_masala", "quantities_g": "250,30,40,100,120,15,15,20,8", "cuisine": "indian", "menu_type": "veg", "calories": 380, "prep_time_min": 40, "servings": 4},
    {"dish_name": "Aloo Paratha", "ingredients": "wheat,potato,onion,coriander,oil,butter,cumin,turmeric", "quantities_g": "250,200,60,20,15,20,5,3", "cuisine": "indian", "menu_type": "veg", "calories": 320, "prep_time_min": 35, "servings": 4},
    {"dish_name": "Mixed Vegetable Curry", "ingredients": "potato,carrot,peas,cauliflower,onion,tomato,oil,turmeric,garam_masala", "quantities_g": "150,100,80,100,100,80,25,3,8", "cuisine": "indian", "menu_type": "veg", "calories": 200, "prep_time_min": 35, "servings": 4},
    {"dish_name": "Rajma Chawal", "ingredients": "kidney_bean,rice,onion,tomato,garlic,ginger,oil,turmeric,cumin", "quantities_g": "200,300,100,100,15,15,25,3,5", "cuisine": "indian", "menu_type": "veg", "calories": 310, "prep_time_min": 50, "servings": 6},
    {"dish_name": "Matar Paneer", "ingredients": "peas,paneer,onion,tomato,cream,garlic,oil,garam_masala,turmeric", "quantities_g": "200,200,100,100,30,15,25,8,3", "cuisine": "indian", "menu_type": "veg", "calories": 290, "prep_time_min": 35, "servings": 4},
    {"dish_name": "Bhindi Masala", "ingredients": "okra,onion,tomato,oil,turmeric,cumin,coriander,amchur", "quantities_g": "300,100,80,25,3,5,5,3", "cuisine": "indian", "menu_type": "veg", "calories": 150, "prep_time_min": 25, "servings": 4},
    {"dish_name": "Sambar", "ingredients": "lentils,drumstick,tomato,onion,carrot,tamarind,oil,mustard,turmeric", "quantities_g": "150,200,100,80,80,20,20,5,3", "cuisine": "south_indian", "menu_type": "veg", "calories": 180, "prep_time_min": 45, "servings": 6},
    {"dish_name": "Idli Sambar", "ingredients": "rice,lentils,fenugreek,drumstick,tomato,oil,mustard,turmeric,salt", "quantities_g": "300,100,10,150,80,15,5,3,5", "cuisine": "south_indian", "menu_type": "vegan", "calories": 220, "prep_time_min": 60, "servings": 6},
    {"dish_name": "Masala Dosa", "ingredients": "rice,lentils,potato,onion,oil,mustard,turmeric,coriander,ginger", "quantities_g": "300,100,200,100,30,5,3,10,10", "cuisine": "south_indian", "menu_type": "vegan", "calories": 300, "prep_time_min": 50, "servings": 4},
    {"dish_name": "Upma", "ingredients": "semolina,onion,tomato,carrot,oil,mustard,turmeric,ginger,peanut", "quantities_g": "200,100,60,60,20,5,3,10,30", "cuisine": "south_indian", "menu_type": "vegan", "calories": 250, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Poha", "ingredients": "flattened_rice,onion,potato,peanut,oil,mustard,turmeric,coriander,lemon", "quantities_g": "200,80,100,30,20,5,3,10,15", "cuisine": "indian", "menu_type": "vegan", "calories": 240, "prep_time_min": 15, "servings": 4},
    {"dish_name": "Khichdi", "ingredients": "rice,lentils,potato,turmeric,ghee,cumin,ginger,salt", "quantities_g": "200,100,150,3,20,5,10,5", "cuisine": "indian", "menu_type": "veg", "calories": 280, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Methi Thepla", "ingredients": "wheat,fenugreek,yogurt,oil,turmeric,cumin,chili,salt", "quantities_g": "250,100,60,20,3,5,5,5", "cuisine": "gujarati", "menu_type": "veg", "calories": 270, "prep_time_min": 25, "servings": 4},
    {"dish_name": "Pav Bhaji", "ingredients": "potato,tomato,peas,onion,butter,bread,pav_bhaji_masala,coriander", "quantities_g": "300,200,100,100,40,200,15,20", "cuisine": "indian", "menu_type": "veg", "calories": 380, "prep_time_min": 35, "servings": 6},
    {"dish_name": "Bisi Bele Bath", "ingredients": "rice,lentils,vegetable,tamarind,oil,ghee,mustard,turmeric,garam_masala", "quantities_g": "200,100,200,20,20,15,5,3,8", "cuisine": "south_indian", "menu_type": "vegan", "calories": 290, "prep_time_min": 55, "servings": 6},
    {"dish_name": "Dum Aloo", "ingredients": "potato,yogurt,onion,tomato,cream,oil,garam_masala,turmeric,coriander", "quantities_g": "400,80,100,100,30,30,10,3,10", "cuisine": "indian", "menu_type": "veg", "calories": 310, "prep_time_min": 45, "servings": 4},
    {"dish_name": "Avial", "ingredients": "carrot,drumstick,yam,coconut,yogurt,turmeric,oil,mustard,curry_leaf", "quantities_g": "100,150,100,80,60,3,15,5,5", "cuisine": "south_indian", "menu_type": "veg", "calories": 190, "prep_time_min": 40, "servings": 4},
    {"dish_name": "Lemon Rice", "ingredients": "rice,lemon,peanut,oil,mustard,turmeric,curry_leaf,chili,salt", "quantities_g": "300,40,50,20,5,3,5,5,5", "cuisine": "south_indian", "menu_type": "vegan", "calories": 320, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Curd Rice", "ingredients": "rice,yogurt,milk,ginger,oil,mustard,curry_leaf,pomegranate,salt", "quantities_g": "300,200,50,10,15,5,5,30,5", "cuisine": "south_indian", "menu_type": "veg", "calories": 290, "prep_time_min": 15, "servings": 4},
    {"dish_name": "Baingan Bharta", "ingredients": "eggplant,onion,tomato,garlic,oil,turmeric,coriander,cumin,chili", "quantities_g": "400,100,100,15,25,3,10,5,5", "cuisine": "punjabi", "menu_type": "vegan", "calories": 180, "prep_time_min": 40, "servings": 4},
    {"dish_name": "Sarson Ka Saag", "ingredients": "mustard_greens,spinach,onion,garlic,ginger,oil,butter,makki_atta,salt", "quantities_g": "400,100,80,15,15,20,30,50,5", "cuisine": "punjabi", "menu_type": "veg", "calories": 200, "prep_time_min": 50, "servings": 4},
    {"dish_name": "Chole Bhature", "ingredients": "chickpea,flour,yogurt,onion,tomato,oil,garam_masala,baking_soda,cumin", "quantities_g": "250,250,60,120,100,40,10,5,5", "cuisine": "punjabi", "menu_type": "veg", "calories": 420, "prep_time_min": 50, "servings": 4},
    {"dish_name": "Dal Makhani", "ingredients": "black_lentil,kidney_bean,butter,cream,tomato,onion,garlic,ginger,oil", "quantities_g": "200,50,40,30,120,100,15,15,20", "cuisine": "punjabi", "menu_type": "veg", "calories": 340, "prep_time_min": 60, "servings": 6},
    {"dish_name": "Aloo Matar", "ingredients": "potato,peas,onion,tomato,oil,turmeric,garam_masala,cumin,coriander", "quantities_g": "250,150,100,80,25,3,8,5,5", "cuisine": "indian", "menu_type": "vegan", "calories": 220, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Gobhi Manchurian", "ingredients": "cauliflower,flour,onion,garlic,ginger,soy_sauce,tomato_ketchup,oil,cornflour", "quantities_g": "400,80,100,15,15,20,30,40,20", "cuisine": "indo_chinese", "menu_type": "vegan", "calories": 280, "prep_time_min": 35, "servings": 4},

    # Non-Veg Indian
    {"dish_name": "Chicken Curry", "ingredients": "chicken,onion,tomato,garlic,ginger,oil,turmeric,garam_masala,yogurt", "quantities_g": "500,150,120,20,20,30,3,10,80", "cuisine": "indian", "menu_type": "non-veg", "calories": 380, "prep_time_min": 50, "servings": 4},
    {"dish_name": "Butter Chicken", "ingredients": "chicken,butter,cream,tomato,onion,garlic,ginger,garam_masala,oil", "quantities_g": "500,40,60,150,100,20,20,10,20", "cuisine": "punjabi", "menu_type": "non-veg", "calories": 430, "prep_time_min": 55, "servings": 4},
    {"dish_name": "Mutton Biryani", "ingredients": "rice,mutton,onion,tomato,yogurt,ghee,garam_masala,saffron,oil", "quantities_g": "400,400,200,100,100,40,15,1,30", "cuisine": "hyderabadi", "menu_type": "non-veg", "calories": 520, "prep_time_min": 90, "servings": 6},
    {"dish_name": "Chicken Biryani", "ingredients": "rice,chicken,onion,tomato,yogurt,ghee,garam_masala,oil,coriander", "quantities_g": "400,400,200,100,100,40,15,30,20", "cuisine": "hyderabadi", "menu_type": "non-veg", "calories": 480, "prep_time_min": 75, "servings": 6},
    {"dish_name": "Fish Fry", "ingredients": "fish,lemon,garlic,ginger,turmeric,chili,oil,coriander,salt", "quantities_g": "500,30,15,15,3,5,40,10,5", "cuisine": "coastal", "menu_type": "non-veg", "calories": 280, "prep_time_min": 25, "servings": 4},
    {"dish_name": "Egg Curry", "ingredients": "egg,onion,tomato,garlic,ginger,oil,turmeric,garam_masala,coriander", "quantities_g": "300,150,120,15,15,25,3,8,10", "cuisine": "indian", "menu_type": "non-veg", "calories": 320, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Keema Matar", "ingredients": "mutton_mince,peas,onion,tomato,garlic,ginger,oil,garam_masala,turmeric", "quantities_g": "400,150,150,120,15,15,30,10,3", "cuisine": "indian", "menu_type": "non-veg", "calories": 400, "prep_time_min": 40, "servings": 4},
    {"dish_name": "Prawn Masala", "ingredients": "prawn,onion,tomato,coconut,garlic,ginger,oil,turmeric,garam_masala", "quantities_g": "400,150,120,80,15,15,25,3,8", "cuisine": "coastal", "menu_type": "non-veg", "calories": 310, "prep_time_min": 35, "servings": 4},
    {"dish_name": "Chicken Tikka Masala", "ingredients": "chicken,yogurt,cream,tomato,onion,garlic,ginger,garam_masala,oil", "quantities_g": "500,80,40,150,120,20,20,12,30", "cuisine": "punjabi", "menu_type": "non-veg", "calories": 450, "prep_time_min": 60, "servings": 4},
    {"dish_name": "Mutton Rogan Josh", "ingredients": "mutton,yogurt,onion,garlic,ginger,oil,kashmiri_chili,garam_masala,turmeric", "quantities_g": "500,100,150,20,20,35,8,12,3", "cuisine": "kashmiri", "menu_type": "non-veg", "calories": 470, "prep_time_min": 75, "servings": 4},
    {"dish_name": "Egg Bhurji", "ingredients": "egg,onion,tomato,chili,oil,turmeric,coriander,salt,ginger", "quantities_g": "300,100,80,10,20,2,10,5,8", "cuisine": "indian", "menu_type": "non-veg", "calories": 250, "prep_time_min": 15, "servings": 3},
    {"dish_name": "Chicken Pulao", "ingredients": "rice,chicken,onion,carrot,peas,oil,garam_masala,ginger,garlic", "quantities_g": "300,350,150,80,80,30,10,15,15", "cuisine": "indian", "menu_type": "non-veg", "calories": 420, "prep_time_min": 60, "servings": 5},
    {"dish_name": "Fish Curry", "ingredients": "fish,coconut,tomato,onion,garlic,turmeric,oil,tamarind,mustard", "quantities_g": "500,100,120,150,20,3,25,20,5", "cuisine": "coastal", "menu_type": "non-veg", "calories": 300, "prep_time_min": 35, "servings": 4},

    # Continental / Global
    {"dish_name": "Pasta Arrabiata", "ingredients": "pasta,tomato,garlic,olive_oil,chili,basil,salt,parmesan", "quantities_g": "300,200,20,30,5,10,5,30", "cuisine": "italian", "menu_type": "vegan", "calories": 380, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Vegetable Fried Rice", "ingredients": "rice,carrot,peas,onion,garlic,soy_sauce,oil,egg,spring_onion", "quantities_g": "300,80,80,80,15,20,25,100,30", "cuisine": "chinese", "menu_type": "non-veg", "calories": 340, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Veg Fried Rice", "ingredients": "rice,carrot,peas,onion,garlic,soy_sauce,oil,spring_onion,capsicum", "quantities_g": "300,80,80,80,15,20,25,30,60", "cuisine": "chinese", "menu_type": "vegan", "calories": 300, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Noodles Stir Fry", "ingredients": "noodles,carrot,capsicum,onion,garlic,soy_sauce,oil,spring_onion,cabbage", "quantities_g": "250,80,80,80,15,20,25,30,80", "cuisine": "chinese", "menu_type": "vegan", "calories": 320, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Veggie Burger", "ingredients": "bread,potato,onion,carrot,lettuce,tomato,cheese,butter,oil", "quantities_g": "160,200,60,60,40,60,40,20,15", "cuisine": "american", "menu_type": "veg", "calories": 420, "prep_time_min": 25, "servings": 2},
    {"dish_name": "Chicken Sandwich", "ingredients": "bread,chicken,lettuce,tomato,cheese,mayonnaise,butter,onion,mustard", "quantities_g": "160,200,40,60,40,30,20,40,10", "cuisine": "american", "menu_type": "non-veg", "calories": 450, "prep_time_min": 20, "servings": 2},
    {"dish_name": "Greek Salad", "ingredients": "tomato,cucumber,onion,olive,feta_cheese,olive_oil,oregano,lemon,salt", "quantities_g": "200,150,60,50,80,30,3,20,5", "cuisine": "mediterranean", "menu_type": "veg", "calories": 190, "prep_time_min": 10, "servings": 2},
    {"dish_name": "Tomato Soup", "ingredients": "tomato,onion,garlic,butter,cream,vegetable_stock,basil,salt,pepper", "quantities_g": "400,100,15,20,30,200,10,5,3", "cuisine": "continental", "menu_type": "veg", "calories": 150, "prep_time_min": 25, "servings": 4},
    {"dish_name": "Mushroom Risotto", "ingredients": "rice,mushroom,onion,garlic,butter,cream,parmesan,vegetable_stock,oil", "quantities_g": "300,200,100,15,30,40,40,400,15", "cuisine": "italian", "menu_type": "veg", "calories": 390, "prep_time_min": 40, "servings": 4},
    {"dish_name": "Vegetable Pizza", "ingredients": "flour,tomato,cheese,capsicum,onion,mushroom,olive,oil,yeast", "quantities_g": "300,120,150,80,80,80,50,20,5", "cuisine": "italian", "menu_type": "veg", "calories": 450, "prep_time_min": 50, "servings": 4},
    {"dish_name": "Quesadilla", "ingredients": "flour,cheese,capsicum,onion,sour_cream,tomato,oil,cumin,chili", "quantities_g": "200,100,80,60,40,80,15,5,5", "cuisine": "mexican", "menu_type": "veg", "calories": 380, "prep_time_min": 15, "servings": 2},
    {"dish_name": "Bean Burrito", "ingredients": "flour,kidney_bean,rice,cheese,tomato,onion,sour_cream,oil,cumin", "quantities_g": "160,150,100,60,80,60,40,15,5", "cuisine": "mexican", "menu_type": "veg", "calories": 420, "prep_time_min": 20, "servings": 2},
    {"dish_name": "Veggie Stir Fry", "ingredients": "broccoli,carrot,capsicum,onion,garlic,soy_sauce,oil,ginger,cornflour", "quantities_g": "200,100,100,80,15,20,25,10,10", "cuisine": "chinese", "menu_type": "vegan", "calories": 180, "prep_time_min": 15, "servings": 3},
    {"dish_name": "Lentil Soup", "ingredients": "lentils,carrot,onion,garlic,tomato,vegetable_stock,oil,cumin,turmeric", "quantities_g": "200,100,100,15,100,400,20,5,3", "cuisine": "continental", "menu_type": "vegan", "calories": 200, "prep_time_min": 35, "servings": 6},
    {"dish_name": "Oatmeal Porridge", "ingredients": "oats,milk,banana,honey,cinnamon,salt", "quantities_g": "100,250,100,20,3,2", "cuisine": "western", "menu_type": "veg", "calories": 310, "prep_time_min": 10, "servings": 2},
    {"dish_name": "French Toast", "ingredients": "bread,egg,milk,sugar,butter,cinnamon,salt,oil", "quantities_g": "160,200,100,20,20,3,2,15", "cuisine": "western", "menu_type": "non-veg", "calories": 360, "prep_time_min": 15, "servings": 2},
    {"dish_name": "Pancakes", "ingredients": "flour,egg,milk,butter,sugar,baking_powder,salt,oil", "quantities_g": "200,100,200,30,30,5,3,15", "cuisine": "american", "menu_type": "non-veg", "calories": 380, "prep_time_min": 20, "servings": 3},
    {"dish_name": "Hummus with Pita", "ingredients": "chickpea,tahini,lemon,garlic,olive_oil,bread,salt,cumin,paprika", "quantities_g": "250,60,40,15,30,200,5,3,2", "cuisine": "middle_eastern", "menu_type": "vegan", "calories": 350, "prep_time_min": 15, "servings": 4},
    {"dish_name": "Falafel", "ingredients": "chickpea,parsley,garlic,onion,cumin,coriander,flour,oil,salt", "quantities_g": "300,30,20,80,8,8,50,200,5", "cuisine": "middle_eastern", "menu_type": "vegan", "calories": 330, "prep_time_min": 30, "servings": 4},

    # Rice / Bread specials
    {"dish_name": "Plain Rice & Dal", "ingredients": "rice,lentils,turmeric,ghee,cumin,salt,water", "quantities_g": "300,150,3,15,5,5,600", "cuisine": "indian", "menu_type": "vegan", "calories": 340, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Jeera Rice", "ingredients": "rice,cumin,ghee,oil,salt,bay_leaf", "quantities_g": "300,10,15,10,5,2", "cuisine": "indian", "menu_type": "vegan", "calories": 290, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Tomato Rice", "ingredients": "rice,tomato,onion,oil,mustard,turmeric,curry_leaf,chili,salt", "quantities_g": "300,150,80,20,5,3,5,5,5", "cuisine": "south_indian", "menu_type": "vegan", "calories": 300, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Pulao", "ingredients": "rice,carrot,peas,onion,oil,bay_leaf,clove,cumin,salt", "quantities_g": "300,80,80,100,25,2,3,5,5", "cuisine": "indian", "menu_type": "vegan", "calories": 310, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Egg Fried Rice", "ingredients": "rice,egg,onion,carrot,peas,soy_sauce,oil,spring_onion,garlic", "quantities_g": "300,200,80,60,60,20,25,30,15", "cuisine": "chinese", "menu_type": "non-veg", "calories": 350, "prep_time_min": 20, "servings": 4},
    {"dish_name": "Chapati & Sabzi", "ingredients": "wheat,potato,onion,tomato,oil,turmeric,cumin,coriander,salt", "quantities_g": "250,200,100,80,20,3,5,10,5", "cuisine": "indian", "menu_type": "vegan", "calories": 300, "prep_time_min": 25, "servings": 4},

    # Vegan extras
    {"dish_name": "Tofu Scramble", "ingredients": "tofu,onion,capsicum,tomato,turmeric,oil,garlic,salt,pepper", "quantities_g": "300,80,80,80,3,20,15,5,3", "cuisine": "western", "menu_type": "vegan", "calories": 220, "prep_time_min": 15, "servings": 3},
    {"dish_name": "Banana Smoothie Bowl", "ingredients": "banana,oats,almond_milk,honey,chia_seed,strawberry,blueberry,coconut_flakes", "quantities_g": "200,80,200,20,15,60,60,20", "cuisine": "western", "menu_type": "vegan", "calories": 350, "prep_time_min": 10, "servings": 2},
    {"dish_name": "Vegetable Soup", "ingredients": "carrot,potato,onion,tomato,celery,garlic,oil,salt,pepper", "quantities_g": "100,100,80,100,60,15,15,5,3", "cuisine": "continental", "menu_type": "vegan", "calories": 120, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Mushroom Stir Fry", "ingredients": "mushroom,onion,garlic,soy_sauce,oil,ginger,spring_onion,chili,cornflour", "quantities_g": "300,100,15,25,25,10,30,5,10", "cuisine": "chinese", "menu_type": "vegan", "calories": 160, "prep_time_min": 15, "servings": 3},
    {"dish_name": "Kadai Vegetable", "ingredients": "capsicum,onion,tomato,paneer,oil,turmeric,coriander,garam_masala,garlic", "quantities_g": "150,100,100,150,25,3,8,8,15", "cuisine": "indian", "menu_type": "veg", "calories": 260, "prep_time_min": 30, "servings": 4},
    {"dish_name": "Pesarattu", "ingredients": "green_moong,ginger,chili,onion,oil,salt,cumin,coriander", "quantities_g": "250,15,10,80,20,5,5,10", "cuisine": "andhra", "menu_type": "vegan", "calories": 240, "prep_time_min": 20, "servings": 4},
]

SYNTHETIC_NUTRITION = [
    {"ingredient": "rice", "calories_per_100g": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3, "shelf_life_days": 365},
    {"ingredient": "wheat", "calories_per_100g": 340, "protein_g": 13, "carbs_g": 71, "fat_g": 2.5, "shelf_life_days": 365},
    {"ingredient": "flour", "calories_per_100g": 364, "protein_g": 10, "carbs_g": 76, "fat_g": 1, "shelf_life_days": 180},
    {"ingredient": "semolina", "calories_per_100g": 360, "protein_g": 13, "carbs_g": 72, "fat_g": 1, "shelf_life_days": 180},
    {"ingredient": "oats", "calories_per_100g": 389, "protein_g": 17, "carbs_g": 66, "fat_g": 7, "shelf_life_days": 180},
    {"ingredient": "lentils", "calories_per_100g": 116, "protein_g": 9, "carbs_g": 20, "fat_g": 0.4, "shelf_life_days": 365},
    {"ingredient": "chickpea", "calories_per_100g": 164, "protein_g": 9, "carbs_g": 27, "fat_g": 2.6, "shelf_life_days": 365},
    {"ingredient": "kidney_bean", "calories_per_100g": 127, "protein_g": 8.7, "carbs_g": 22.8, "fat_g": 0.5, "shelf_life_days": 365},
    {"ingredient": "black_lentil", "calories_per_100g": 116, "protein_g": 9, "carbs_g": 20, "fat_g": 0.4, "shelf_life_days": 365},
    {"ingredient": "potato", "calories_per_100g": 77, "protein_g": 2, "carbs_g": 17, "fat_g": 0.1, "shelf_life_days": 14},
    {"ingredient": "tomato", "calories_per_100g": 18, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2, "shelf_life_days": 7},
    {"ingredient": "onion", "calories_per_100g": 40, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.1, "shelf_life_days": 30},
    {"ingredient": "garlic", "calories_per_100g": 149, "protein_g": 6.4, "carbs_g": 33, "fat_g": 0.5, "shelf_life_days": 90},
    {"ingredient": "ginger", "calories_per_100g": 80, "protein_g": 1.8, "carbs_g": 18, "fat_g": 0.8, "shelf_life_days": 14},
    {"ingredient": "spinach", "calories_per_100g": 23, "protein_g": 2.9, "carbs_g": 3.6, "fat_g": 0.4, "shelf_life_days": 5},
    {"ingredient": "cauliflower", "calories_per_100g": 25, "protein_g": 1.9, "carbs_g": 5, "fat_g": 0.3, "shelf_life_days": 7},
    {"ingredient": "carrot", "calories_per_100g": 41, "protein_g": 0.9, "carbs_g": 10, "fat_g": 0.2, "shelf_life_days": 21},
    {"ingredient": "peas", "calories_per_100g": 81, "protein_g": 5.4, "carbs_g": 14, "fat_g": 0.4, "shelf_life_days": 5},
    {"ingredient": "capsicum", "calories_per_100g": 31, "protein_g": 1, "carbs_g": 6, "fat_g": 0.3, "shelf_life_days": 7},
    {"ingredient": "mushroom", "calories_per_100g": 22, "protein_g": 3.1, "carbs_g": 3.3, "fat_g": 0.3, "shelf_life_days": 5},
    {"ingredient": "eggplant", "calories_per_100g": 25, "protein_g": 1, "carbs_g": 6, "fat_g": 0.2, "shelf_life_days": 7},
    {"ingredient": "okra", "calories_per_100g": 33, "protein_g": 1.9, "carbs_g": 7.5, "fat_g": 0.2, "shelf_life_days": 5},
    {"ingredient": "broccoli", "calories_per_100g": 34, "protein_g": 2.8, "carbs_g": 7, "fat_g": 0.4, "shelf_life_days": 7},
    {"ingredient": "paneer", "calories_per_100g": 265, "protein_g": 18, "carbs_g": 3.4, "fat_g": 20, "shelf_life_days": 5},
    {"ingredient": "yogurt", "calories_per_100g": 59, "protein_g": 10, "carbs_g": 3.6, "fat_g": 0.4, "shelf_life_days": 7},
    {"ingredient": "milk", "calories_per_100g": 61, "protein_g": 3.2, "carbs_g": 4.8, "fat_g": 3.3, "shelf_life_days": 5},
    {"ingredient": "cheese", "calories_per_100g": 402, "protein_g": 25, "carbs_g": 1.3, "fat_g": 33, "shelf_life_days": 14},
    {"ingredient": "butter", "calories_per_100g": 717, "protein_g": 0.9, "carbs_g": 0.1, "fat_g": 81, "shelf_life_days": 30},
    {"ingredient": "cream", "calories_per_100g": 340, "protein_g": 2.1, "carbs_g": 2.8, "fat_g": 36, "shelf_life_days": 7},
    {"ingredient": "ghee", "calories_per_100g": 900, "protein_g": 0, "carbs_g": 0, "fat_g": 100, "shelf_life_days": 180},
    {"ingredient": "oil", "calories_per_100g": 884, "protein_g": 0, "carbs_g": 0, "fat_g": 100, "shelf_life_days": 365},
    {"ingredient": "olive_oil", "calories_per_100g": 884, "protein_g": 0, "carbs_g": 0, "fat_g": 100, "shelf_life_days": 365},
    {"ingredient": "egg", "calories_per_100g": 143, "protein_g": 13, "carbs_g": 1.1, "fat_g": 9.5, "shelf_life_days": 21},
    {"ingredient": "chicken", "calories_per_100g": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6, "shelf_life_days": 3},
    {"ingredient": "mutton", "calories_per_100g": 258, "protein_g": 25, "carbs_g": 0, "fat_g": 17, "shelf_life_days": 3},
    {"ingredient": "fish", "calories_per_100g": 206, "protein_g": 22, "carbs_g": 0, "fat_g": 12, "shelf_life_days": 2},
    {"ingredient": "prawn", "calories_per_100g": 99, "protein_g": 24, "carbs_g": 0.2, "fat_g": 0.3, "shelf_life_days": 2},
    {"ingredient": "tofu", "calories_per_100g": 76, "protein_g": 8, "carbs_g": 1.9, "fat_g": 4.8, "shelf_life_days": 5},
    {"ingredient": "peanut", "calories_per_100g": 567, "protein_g": 26, "carbs_g": 16, "fat_g": 49, "shelf_life_days": 180},
    {"ingredient": "cashew", "calories_per_100g": 553, "protein_g": 18, "carbs_g": 30, "fat_g": 44, "shelf_life_days": 180},
    {"ingredient": "lemon", "calories_per_100g": 29, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.3, "shelf_life_days": 14},
    {"ingredient": "banana", "calories_per_100g": 89, "protein_g": 1.1, "carbs_g": 23, "fat_g": 0.3, "shelf_life_days": 5},
    {"ingredient": "coconut", "calories_per_100g": 354, "protein_g": 3.3, "carbs_g": 15, "fat_g": 33, "shelf_life_days": 7},
    {"ingredient": "soy_sauce", "calories_per_100g": 53, "protein_g": 8, "carbs_g": 5, "fat_g": 0.1, "shelf_life_days": 730},
    {"ingredient": "turmeric", "calories_per_100g": 312, "protein_g": 9.7, "carbs_g": 68, "fat_g": 3.3, "shelf_life_days": 730},
    {"ingredient": "cumin", "calories_per_100g": 375, "protein_g": 18, "carbs_g": 44, "fat_g": 22, "shelf_life_days": 730},
    {"ingredient": "coriander", "calories_per_100g": 23, "protein_g": 2.1, "carbs_g": 3.7, "fat_g": 0.5, "shelf_life_days": 5},
    {"ingredient": "garam_masala", "calories_per_100g": 350, "protein_g": 13, "carbs_g": 55, "fat_g": 10, "shelf_life_days": 730},
    {"ingredient": "pasta", "calories_per_100g": 371, "protein_g": 13, "carbs_g": 75, "fat_g": 1.5, "shelf_life_days": 730},
    {"ingredient": "noodles", "calories_per_100g": 138, "protein_g": 4.5, "carbs_g": 25, "fat_g": 2.1, "shelf_life_days": 730},
    {"ingredient": "bread", "calories_per_100g": 265, "protein_g": 9, "carbs_g": 49, "fat_g": 3.2, "shelf_life_days": 5},
    {"ingredient": "flattened_rice", "calories_per_100g": 110, "protein_g": 2.5, "carbs_g": 23, "fat_g": 0.3, "shelf_life_days": 180},
]


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_dishes_csv(filepath: str) -> pd.DataFrame:
    """Generate and save the synthetic dishes dataset."""
    df = pd.DataFrame(SYNTHETIC_DISHES)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  ✅ Generated {len(df)} dishes → {filepath}")
    return df


def generate_nutrition_csv(filepath: str) -> pd.DataFrame:
    """Generate and save the ingredient nutrition dataset."""
    df = pd.DataFrame(SYNTHETIC_NUTRITION)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  ✅ Generated {len(df)} ingredient nutrition records → {filepath}")
    return df


def load_or_generate_dishes(filepath: str = None) -> pd.DataFrame:
    """
    Load dishes dataset from CSV. Auto-generates if file not found.

    Returns:
        pd.DataFrame: Dishes dataset
    """
    if filepath is None:
        filepath = config.DISHES_DATASET_FILE

    if not os.path.exists(filepath):
        print(f"  ⚠️  Dishes dataset not found. Auto-generating synthetic data...")
        df = generate_dishes_csv(filepath)
    else:
        df = pd.read_csv(filepath)
        print(f"  ✅ Loaded {len(df)} dishes from {filepath}")
    return df


def load_or_generate_nutrition(filepath: str = None) -> pd.DataFrame:
    """
    Load ingredient nutrition CSV. Auto-generates if file not found.

    Returns:
        pd.DataFrame: Ingredient nutrition dataset
    """
    if filepath is None:
        filepath = config.INGREDIENT_NUTRITION_FILE

    if not os.path.exists(filepath):
        print(f"  ⚠️  Nutrition data not found. Auto-generating...")
        df = generate_nutrition_csv(filepath)
    else:
        df = pd.read_csv(filepath)
        print(f"  ✅ Loaded {len(df)} ingredient records from {filepath}")
    return df


def parse_ingredients(ingredient_str: str) -> list:
    """Parse comma-separated ingredient string into a list."""
    return [ing.strip().lower() for ing in ingredient_str.split(',')]


def parse_quantities(quantities_str: str) -> list:
    """Parse comma-separated quantity string into a list of floats."""
    try:
        return [float(q.strip()) for q in quantities_str.split(',')]
    except Exception:
        return []


def build_ingredient_matrix(dishes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a multi-hot encoded matrix: rows = dishes, columns = ingredients.

    Returns:
        pd.DataFrame: Multi-hot ingredient matrix (index = dish_name)
    """
    # Collect all unique ingredients
    all_ingredients = set()
    for ing_str in dishes_df['ingredients']:
        all_ingredients.update(parse_ingredients(ing_str))
    all_ingredients = sorted(all_ingredients)

    # Build matrix
    rows = []
    for _, row in dishes_df.iterrows():
        dish_ings = set(parse_ingredients(row['ingredients']))
        encoded = {ing: (1 if ing in dish_ings else 0) for ing in all_ingredients}
        rows.append(encoded)

    matrix = pd.DataFrame(rows, index=dishes_df['dish_name'])
    return matrix


def compute_dish_complexity(dishes_df: pd.DataFrame) -> pd.Series:
    """
    Compute dish complexity score (0–1) based on number of ingredients and prep time.
    """
    ingredient_counts = dishes_df['ingredients'].apply(
        lambda x: len(parse_ingredients(x))
    )
    prep_times = dishes_df['prep_time_min'].fillna(30)

    # Normalise each component
    ic_norm = (ingredient_counts - ingredient_counts.min()) / (ingredient_counts.max() - ingredient_counts.min() + 1e-9)
    pt_norm = (prep_times - prep_times.min()) / (prep_times.max() - prep_times.min() + 1e-9)

    complexity = 0.5 * ic_norm + 0.5 * pt_norm
    return complexity.round(4)


def compute_estimated_cost(dishes_df: pd.DataFrame) -> pd.Series:
    """
    Estimate relative cost score (0–1) from ingredient quantities and calories.
    Higher calories + quantities → higher estimated cost.
    """
    total_qty = dishes_df['quantities_g'].apply(
        lambda x: sum(parse_quantities(x))
    )
    calories = dishes_df['calories'].fillna(300)

    score = 0.6 * total_qty + 0.4 * calories
    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score_norm.round(4)


def compute_dish_features(dishes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered feature columns to the dish DataFrame.

    Added columns:
        - ingredient_count
        - complexity_score
        - cost_score
        - caloric_density  (calories / total_qty_g)
        - ingredient_list  (parsed list)
    """
    df = dishes_df.copy()

    df['ingredient_list'] = df['ingredients'].apply(parse_ingredients)
    df['ingredient_count'] = df['ingredient_list'].apply(len)
    df['complexity_score'] = compute_dish_complexity(df)
    df['cost_score'] = compute_estimated_cost(df)

    total_qty = df['quantities_g'].apply(lambda x: sum(parse_quantities(x)))
    df['caloric_density'] = (df['calories'] / (total_qty + 1e-9)).round(4)

    return df
