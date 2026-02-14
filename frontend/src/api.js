import axios from 'axios';

const API_BASE_URL = '/api';

export const predictTicket = async (ticketText) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      ticket_text: ticketText,
    });
    return response.data;
  } catch (error) {
    console.error('Error predicting ticket:', error);
    throw error;
  }
};

export const getAllTickets = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/tickets`);
    return response.data;
  } catch (error) {
    console.error('Error fetching tickets:', error);
    throw error;
  }
};
