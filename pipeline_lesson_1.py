# Python Function-Based Pipelines

def load_data():
    marks =[20, 40, 50, 60, 70, 80, 90]
    return marks

def clear_data(data):
    return [x for x in data if x>=50]

def transform_data(data):
    return [x + 5 for x in data]

def analyze_data(data):
    average = sum(data) / len(data)
    return average

#Create pipeline
def run_pipeline():
    data = load_data()
    cleaned = clear_data(data)
    transformed = transform_data(cleaned)
    result = analyze_data(transformed)
    return result

output = run_pipeline()

print("Average marks" , output)