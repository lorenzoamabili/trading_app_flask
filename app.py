import os
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime, timedelta
import plotly.io as pio
# Assuming this imports the function that returns the plotly figure
from main import stock_analysis

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random key for flash messages

# Serverless function entry point


@app.route('/', methods=['GET', 'POST'])
def handler():
    if request.method == 'POST':
        # Get user inputs from the form
        stock = request.form.get('stock_symbol', 'AAPL')
        months = int(request.form.get('months', 6))
        buy_date_str = request.form.get(
            'buy_date', datetime.today().strftime('%Y-%m-%d'))
        buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d')

        # Calculate the allowed date range
        max_date = datetime.today()
        min_date = max_date - timedelta(days=months * 30)

        # Ensure the buy date is within the range
        if buy_date > max_date:
            flash("Buy Date cannot be in the future.", 'warning')
            return redirect(url_for('handler'))

        if buy_date < min_date:
            flash(f"Buy Date cannot be earlier than {
                  min_date.strftime('%Y-%m-%d')}.", 'warning')
            return redirect(url_for('handler'))

        # Generate the stock plot using the stock_analysis function
        fig = stock_analysis(stock, buy_date.strftime(
            '%Y-%m-%d %H:%M:%S'), months)
        graph_html = pio.to_html(fig, full_html=False)

        return render_template('index.html', graph_html=graph_html)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
