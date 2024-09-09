from kafka import KafkaConsumer, KafkaProducer

def consume_stream():
    consumer = KafkaConsumer(
        'content_stream',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='content-group'
    )

    for message in consumer:
        print(f"Received message: {message.value}")

if __name__ == "__main__":
    consume_stream()
