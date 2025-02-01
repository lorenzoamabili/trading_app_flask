from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import yfinance as yf
from main import stock_analysis

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages (optional)

# Function to generate stock analysis plot


def generate_stock_plot(stock_symbol, start_date, months):
    # Fetch historical stock data
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Assuming stock_analysis generates a plot for us
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Stock Close Price')
    ax.set_title(f"{stock_symbol} Stock Analysis")
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)  # Rewind the BytesIO object to the beginning

    return img

# Route for the home page


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user inputs from the form
        stock = request.form.get('stock_symbol', 'AAPL')
        months = int(request.form.get('months', 6))
        buy_date = request.form.get(
            'buy_date', datetime.today().strftime('%Y-%m-%d'))
        buy_date = datetime.strptime(buy_date, '%Y-%m-%d')

        # Calculate the allowed date range
        max_date = datetime.today()
        min_date = max_date - timedelta(days=months * 30)

        # Ensure the buy date is within the range
        if buy_date > max_date:
            flash("Buy Date cannot be in the future.", 'warning')
            return redirect(url_for('home'))

        if buy_date < min_date:
            flash(f"Buy Date cannot be earlier than {
                  min_date.strftime('%Y-%m-%d')}.", 'warning')
            return redirect(url_for('home'))

        # Check if the selected date is a Saturday or Sunday
        if buy_date.weekday() == 5:  # Saturday
            flash("You cannot select a Saturday. Please choose a weekday.", 'warning')
            buy_date = buy_date - timedelta(days=1)
        elif buy_date.weekday() == 6:  # Sunday
            flash("You cannot select a Sunday. Please choose a weekday.", 'warning')
            buy_date = buy_date - timedelta(days=2)

        fig = stock_analysis(stock, buy_date.strftime(
            '%Y-%m-%d %H:%M:%S'), months)

        return render_template('index.html', graph_html=fig)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
