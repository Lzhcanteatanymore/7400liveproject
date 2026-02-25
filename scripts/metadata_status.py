import pandas as pd
import re

my_data = pd.read_excel("labels_example.xlsx")

#print(my_data.head())
#print(my_data["food_name (lowercase)"].unique())

image_name = r'^[0-9]{8}_[0-9]{3}$'
author = r'^[0-9]{7}$'
food_name = ["apple", "orange", "carrot", "cucumber"]
food_logical = [True, False]
food_num = [1, 2, 3, 4, 5]
light_type = ["L1", "L2", "L3", "L4"]
back_type = ["B1", "B2", "B3", "B4"]

# Condition check
condition = (
    re.fullmatch(image_name, my_data["image_id(key)"]) &
    re.fullmatch(author, my_data["student_id(alphabetical)"]) &
    my_data["food_name (lowercase)"].isin(food_name) &
    my_data["is_fruit(true/false)"].isin(food_logical) &
    my_data["fruit_instance"].isin(food_num) &
    my_data["lighting_session"].isin(light_type) &
    my_data["background_id"].isin(back_type)
)



my_data["metadata_status"] = condition.map({True: "COMPLETE", False: "INCOMPLETE"})


#print(my_data)
my_data.to_csv('labels.csv', index=False)





