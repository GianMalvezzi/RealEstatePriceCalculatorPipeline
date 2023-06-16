# Real Estate Price Calculator

## Project Overview

The Real Estate Price Calculator is a machine learning pipeline that predicts the prices of real estate properties based on various features and attributes. This pipeline aims to assist real estate professionals, buyers, and sellers in estimating property values accurately and making informed decisions.

The pipeline leverages machine learning algorithms to learn patterns and relationships between the features of real estate properties and their corresponding prices. By training the models on historical data, the pipeline can generalize and provide price estimates for new, unseen properties.

This project provides a flexible and customizable solution that can be applied to different real estate markets and property types. The pipeline allows users to input property features, such as location, size, amenities, and other relevant factors, and obtain estimated prices based on the trained models.

## Problem Statement

Accurately determining the price of a real estate property is a complex task that involves considering multiple factors and market dynamics. Manual price estimations may be subjective and prone to human bias. Additionally, real estate markets are constantly evolving, making it challenging to keep up with the latest trends and fluctuations.

The Real Estate Price Calculator aims to address these challenges by leveraging machine learning techniques to provide objective and data-driven price estimations. By analyzing historical data and learning from patterns in the market, the pipeline can generate reliable price predictions that align with current market conditions.

The key objectives of the Real Estate Price Calculator are:

- Develop a robust and accurate machine learning pipeline for real estate price estimation.
- Incorporate relevant features and attributes that impact property prices, such as location, size, amenities, and market trends.
- Train and validate machine learning models using historical data to ensure reliable predictions.
- Provide users with an intuitive interface to input property details and obtain estimated prices.
- Continuously update and refine the pipeline based on new data and market insights.

By leveraging the Real Estate Price Calculator, users can gain valuable insights into property valuations, optimize listing prices, negotiate better deals, and make informed investment decisions.

## Column Translation Schema

To provide clarity on the column names and their corresponding meanings used in the Real Estate Price Calculator, the following translation schema is applied:

| Column Name          | Translation           |
| -------------------- | --------------------- |
| ID                   | Property ID           |
| Categoria            | Property category     |
| Tipo                 | Property type         |
| Preco                | Property price        |
| Bairro               | Neighborhood          |
| Cidade               | City                  |
| Condominio           | Condominium fee       |
| IPTU                 | Property tax (IPTU)   |
| Quartos              | Number of bedrooms    |
| Banheiros            | Number of bathrooms   |
| Vagas_carro          | Number of parking spaces |
| Tamanho              | Property size (in sq. m) |
| Data                 | Date of data collection |
| json                 | JSON data             |
| Tempo_Data_Coleta    | Time of data collection |