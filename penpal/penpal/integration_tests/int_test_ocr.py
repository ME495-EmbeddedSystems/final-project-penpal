"""Integration test for the Gemini OCR + QA service."""
import json
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger


class BoardClient(Node):
    """ROS 2 client to request board OCR + QA."""

    def __init__(self):
        """Initialize the client node."""
        super().__init__('board_client')

        self.cli = self.create_client(Trigger, 'read_and_answer_board')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self):
        """Send the request to the OCR + QA service."""
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    """Run the client."""
    rclpy.init()
    client = BoardClient()

    print('\nRequesting board analysis from Gemini...')

    try:
        response = client.send_request()

        if response.success:
            # parse the JSON string into a Python dictionary
            data = json.loads(response.message)

            print('===OUTPUT FROM GEMINI OCR + QA SERVICE===')
            print('=' * 40)
            print('\nTRANSCRIPTION:')
            print('-' * 13)
            print(f'   {data.get('question', 'No text found')}')
            print('=' * 40)
            print('\nRESPONSE:')
            print('-' * 9)
            print(f'   {data.get('answer', 'No answer generated')}')
            print('\n' + '-' * 40 + '\n')
        else:
            print('\n[Error] Service call failed!')
            print(f'Message: {response.message}\n')

    except Exception as e:
        print(f'[Error] Script failed: {e}')
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
