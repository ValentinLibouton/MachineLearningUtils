# Machine Learning Utils

This repository contains common utilities for Machine Learning projects. Here's how you can import and update it in your projects.



## How to import into a project
To add this library as a sub-module to your project, run the following command in the terminal at the root of your project:
```bash
git submodule add git@github.com:ValentinLibouton/MachineLearningUtils.git <path>
```
Replace `<path>` with the relative path in your project where you wish to add the library.



## How to update a project if MachineLearningUtils has been modified

If changes have been made to the MachineLearningUtils library and you wish to retrieve these updates in your project, follow these steps:
1. Open a terminal at the root of your project.
2. Run the following command to update the submodule:
```bash
git submodule update --remote <path>
```
Replace `<path>` with the relative path where the MachineLearningUtils sub-module is located in your project.
3. After updating, don't forget to commit the changes to your project to save the sub-module update:
```bash
git add <path>
git commit -m "MachineLearningUtils updated with the latest changes"
git push
```
