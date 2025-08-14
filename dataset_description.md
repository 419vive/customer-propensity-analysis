# Customer Propensity to Purchase Dataset

## Context
You get many visitors to your website every day, but you know only a small percentage of them are likely to buy from you, while most will perhaps not even return. Right now you may be spending money to re-market to everyone, but perhaps we could use machine learning to identify the most valuable prospects?

## Content
This data set represents a day's worth of visit to a fictional website. Each row represents a unique customer, identified by their unique UserID. The columns represent feature of the users visit (such as the device they were using) and things the user did on the website in that day. These features will be different for every website, but in this data a few of the features we consider are:

- **basket_add_detail**: Did the customer add a product to their shopping basket from the product detail page?
- **sign_in**: Did the customer sign in to the website?
- **saw_homepage**: Did the customer visit the website's homepage?
- **returning_user**: Is this visitor new, or returning?

In this data set we also have a feature showing whether the customer placed an order (**ordered**), which is what we predict on.