import time

def longtime_job():
    print("job start")
    time.sleep(1)
    print('here')
    return "done"

list_job = [longtime_job() for i in range(5)]
print(next(list_job))