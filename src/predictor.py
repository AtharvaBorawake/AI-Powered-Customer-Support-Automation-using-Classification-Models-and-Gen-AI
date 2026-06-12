predictor = TicketClassifier()

queue = predictor.predict(ticket_text)

response = gemini.generate_response(
    ticket_text,
    queue
)