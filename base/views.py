from django.shortcuts import render
from django.conf import settings
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
import pandas as pd
import numpy as np
import io
import os
# import csv
from django.core.files.storage import default_storage

def home(request):
    context = {'message':"Nothing to see here."}
    if request.method == 'POST':
        #print(request.POST)
        # if 'input_file' in request.FILES and 'weights' in request.POST and 'impacts' in request.POST and 'email' in request.POST:
        try:
            input_file = request.FILES.get('input_file')
            weights = request.POST.get('weights')
            impacts = request.POST.get('impacts')
            email = request.POST.get('email')

            input_file_name = default_storage.save('input_file.csv', input_file)
            input_file_path = default_storage.path(input_file_name)

            data = pd.read_csv(input_file_path)#csv.DictReader(input_file.read().decode('utf-8').splitlines())

            # if data.shape[1] < 3:
            #     raise ValueError("Input file must contain three or more columns.")

            data_values = data.iloc[:, 1:]
            # if not all(data_values.map(np.isreal).all()):
            #     raise ValueError("Error: All columns except the first must contain numeric values only.")

            weight = list(map(float, weights.split(',')))
            cost = impacts.split(',')

            if len(weight) != data_values.shape[1]:
                raise ValueError(f"Number of weights ({len(weight)}) does not match the number of criteria ({data_values.shape[1]}).")
            if len(cost) != data_values.shape[1]:
                raise ValueError(f"Number of impacts ({len(cost)}) does not match the number of criteria ({data_values.shape[1]}).")

            if not all(c in ["+", "-"] for c in cost):
                raise ValueError("Impacts must be either '+' (positive) or '-' (negative).")

            normsqrt = np.sqrt((data_values ** 2).sum(axis=0))

            normalised_data = data_values / normsqrt
            if any(normsqrt == 0):
                raise ValueError("One or more columns have zero variance, leading to division by zero during normalization.")

            after_weight_data = normalised_data * weight

            vpos = []
            vneg = []

            for i in range(len(cost)):
                if cost[i] == '+':
                    vpos.append(after_weight_data.iloc[:, i].max())
                    vneg.append(after_weight_data.iloc[:, i].min())
                else:
                    vpos.append(after_weight_data.iloc[:, i].min())
                    vneg.append(after_weight_data.iloc[:, i].max())

            vpos = np.array(vpos)
            vneg = np.array(vneg)

            spos = np.sqrt(((after_weight_data - vpos) ** 2).sum(axis=1))
            sneg = np.sqrt(((after_weight_data - vneg) ** 2).sum(axis=1))

            scores = sneg / (spos + sneg)

            data['Score'] = scores
            data['Rank'] = data['Score'].rank(ascending=False, method='max')

            result_csv = io.StringIO()
            data.to_csv(result_csv, index=False)
            result_csv.seek(0)
            print(result_csv.getvalue())
            email_message=EmailMultiAlternatives('TOPSIS Analysis Results',
                                   None,
                                   settings.DEFAULT_FROM_EMAIL,
                                   [email],
                                   reply_to=[settings.DEFAULT_FROM_EMAIL]
                                   )
            # email_message = EmailMessage(
            #     subject="TOPSIS Analysis Results",
            #     body="Please find the attached CSV file with the TOPSIS analysis results.",
            #     from_email=settings.DEFAULT_FROM_EMAIL,
            #     to=[email],
            # )
            email_message.attach('topsis_results.csv', result_csv.read(), 'text/csv')
            email_message.send()

            context['message'] = 'TOPSIS results have been sent to your email.'
        except FileNotFoundError:
            context['error'] = f"Error: The file '{input_file}' was not found. Please check the file path and try again."
        except ValueError as ve:
            context['error'] = f"ValueError: {ve}"
        except Exception as e:
            context['error'] = f"An unexpected error occurred: {e}"
    print(context)
    return render(request, "index.html", context)
