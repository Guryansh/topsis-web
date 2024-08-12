from django.core.mail import EmailMultiAlternatives
from django.shortcuts import render
from django.conf import settings
import numpy as np
import io
import csv
from django.core.files.storage import default_storage

def home(request):
    context = {'message': "Nothing to see here."}
    if request.method == 'POST':
        try:
            input_file = request.FILES.get('input_file')
            weights = request.POST.get('weights')
            impacts = request.POST.get('impacts')
            email = request.POST.get('email')

            # Save the uploaded file to a temporary location
            input_file_name = default_storage.save('input_file.csv', input_file)
            input_file_path = default_storage.path(input_file_name)

            # Read the CSV file into a list of dictionaries
            data = list(csv.DictReader(open(input_file_path)))

            # Extract numeric data, excluding the first column (assumed to be text)
            data_values = np.array([list(map(float, list(row.values())[1:])) for row in data])

            # Validate the number of columns
            if data_values.shape[1] < 3:
                raise ValueError("Input file must contain three or more columns.")

            # Validate weights and impacts
            weight = list(map(float, weights.split(',')))
            cost = impacts.split(',')

            if len(weight) != data_values.shape[1]:
                raise ValueError(f"Number of weights ({len(weight)}) does not match the number of criteria ({data_values.shape[1]}).")
            if len(cost) != data_values.shape[1]:
                raise ValueError(f"Number of impacts ({len(cost)}) does not match the number of criteria ({data_values.shape[1]}).")
            if not all(c in ["+", "-"] for c in cost):
                raise ValueError("Impacts must be either '+' (positive) or '-' (negative).")

            # Normalize the data
            normsqrt = np.sqrt((data_values ** 2).sum(axis=0))
            if any(normsqrt == 0):
                raise ValueError("One or more columns have zero variance, leading to division by zero during normalization.")
            normalised_data = data_values / normsqrt

            # Apply weights
            after_weight_data = normalised_data * weight

            # Determine ideal and anti-ideal solutions
            vpos = []
            vneg = []

            for i in range(len(cost)):
                if cost[i] == '+':
                    vpos.append(after_weight_data[:, i].max())
                    vneg.append(after_weight_data[:, i].min())
                else:
                    vpos.append(after_weight_data[:, i].min())
                    vneg.append(after_weight_data[:, i].max())

            vpos = np.array(vpos)
            vneg = np.array(vneg)

            # Calculate distances to ideal and anti-ideal solutions
            spos = np.sqrt(((after_weight_data - vpos) ** 2).sum(axis=1))
            sneg = np.sqrt(((after_weight_data - vneg) ** 2).sum(axis=1))

            # Calculate scores and ranks
            scores = sneg / (spos + sneg)
            for i, row in enumerate(data):
                row['Score'] = scores[i]
            data = sorted(data, key=lambda x: x['Score'], reverse=True)
            for i, row in enumerate(data):
                row['Rank'] = i + 1

            # Convert the result to a CSV
            result_csv = io.StringIO()
            writer = csv.DictWriter(result_csv, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            result_csv.seek(0)

            # Send the email with the CSV attachment
            email_message = EmailMultiAlternatives(
                'TOPSIS Analysis Results',
                "Please find the attached CSV file with the TOPSIS analysis results.",
                settings.DEFAULT_FROM_EMAIL,
                [email]
            )
            email_message.attach('topsis_results.csv', result_csv.getvalue(), 'text/csv')
            email_message.send()

            context['message'] = 'TOPSIS results have been sent to your email.'
        except FileNotFoundError:
            context['error'] = f"Error: The file '{input_file}' was not found. Please check the file path and try again."
        except ValueError as ve:
            context['error'] = f"ValueError: {ve}"
        except Exception as e:
            context['error'] = f"An unexpected error occurred: {e}"

    return render(request, "index.html", context)
