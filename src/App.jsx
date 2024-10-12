import React from 'react';
import Hero from './pages/Hero';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SignUp from './pages/SignUp';
import SignIn from './pages/SignIn';
import CaptchaPage from './pages/CaptchaPage';
import Chat from './pages/Chat';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path='/' element={<Hero />} />
        <Route path='/signin' element={<SignIn />} />
        <Route path='/signup' element={<SignUp />} />
        <Route path='/captcha' element={<CaptchaPage />} />
        <Route path='/chat' element={<Chat />} />
      </Routes>
    </Router>
  );
}
