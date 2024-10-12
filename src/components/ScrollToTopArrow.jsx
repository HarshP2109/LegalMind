import React, { useState, useEffect } from 'react';

export default function ScrollToTopArrow() {
  const [showArrow, setShowArrow] = useState(false);

  // Show the arrow when the user scrolls down 300px
  const handleScroll = () => {
    if (window.scrollY > 300) {
      setShowArrow(true);
    } else {
      setShowArrow(false);
    }
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Scroll to the top when the arrow is clicked
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  return (
    <>
      {showArrow && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-16 z-20 right-8 bg-customGreen text-white p-3 rounded-full shadow-lg hover:bg-green-700 transition duration-300"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M5 10l7-7m0 0l7 7m-7-7v18"
            />
          </svg>
        </button>
      )}
    </>
  );
}
