import os 
import pandas as pd
import matplotlib.pyplot as plt
def plot_(col1_single_,ylabel,path_model):
    name=f'{path_model}{ylabel}'
    # Extract the model names and corresponding values
    model_names = list(col1_single_.keys())
    values = list(col1_single_.values())
    metric={0:'MAE'                
            ,1:'MSE',              
            2:'RMSE',               
            3:'R2'       ,          
            4:'Adjusted R2' }
     # Plotting the data for each metric
    fig, axs = plt.subplots(len(metric), 1, figsize=(8, 6 * len(metric)))

    for i, metric_idx in enumerate(metric.keys()):
        axs[i].set_title(metric[metric_idx])

        for j, model in enumerate(model_names):
            axs[i].plot(range(len(values[j])), values[j], label=model)

        axs[i].set_xlabel('Data Point')
        axs[i].set_ylabel(ylabel)
        axs[i].legend(fontsize='small')  # Adjust the legend font size here
        axs[i].set_title(name + ' ' + str(metric[i]))

    plt.tight_layout()
    plt.savefig(name + '_plots.png')
    plt.show()


def plotter(path):
    col1_single_={}
    col2_single_={}
    col3_single_={}
    for dir_ in os.listdir(path):
        path_count=os.path.join(path,dir_)
        print(dir_)
        for path_model in os.listdir(path_count):
            csv_path=os.path.join(path_count,path_model)
            for csv_ in os.listdir(csv_path):
                csv_path=os.path.join(csv_path,csv_)
                
                if csv_=='metrics.csv':
                    data=pd.read_csv(csv_path)
                    name=path_model.split('+')[0]
                    col1_single_[name]=data['ChestGirth(cm)'].tolist()
                    col2_single_[name]=data['HipsGirth(cm)'].tolist()
                    col3_single_[name]=data['WaistGirth(cm)'].tolist()
        
        plot_(col1_single_,'ChestGrith(cm)',dir_)
        plot_(col2_single_,'HipsGirth(cm)',dir_)
        plot_(col3_single_,'WaistGirth(cm)',dir_)
        