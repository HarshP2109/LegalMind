import React,{useState} from 'react'


export default function ContactUs() {
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        message: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log(formData);
        // Add form submission logic here
    };
    return (
        <div className="grid sm:grid-cols-2 px-12">
            <div
                className='flex flex-col sm:items-start items-center justify-center mt-6 sm:mt-0'>
                <h1 className="text-[40px] text-customGreen text-center">Contact Us</h1>
                <p className="w-3/4 hidden sm:block">Need to get in touch with us? Either fill out the form with your inquiry or find the department mail attached below.</p>
            </div>
            <div className="  p-6 ">
                <form onSubmit={handleSubmit}>
                    {/* First Name and Last Name Fields */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div>
                            <label htmlFor="firstName" className="block text-sm font-medium text-gray-700">
                                First name<span className="text-red-500">*</span>
                            </label>
                            <input
                                type="text"
                                name="firstName"
                                value={formData.firstName}
                                onChange={handleChange}
                                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none"
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="lastName" className="block text-sm font-medium text-gray-700">
                                Last name
                            </label>
                            <input
                                type="text"
                                name="lastName"
                                value={formData.lastName}
                                onChange={handleChange}
                                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none"
                            />
                        </div>
                    </div>

                    {/* Email Field */}
                    <div className="mb-4">
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                            Email<span className="text-red-500">*</span>
                        </label>
                        <input
                            type="email"
                            name="email"
                            value={formData.email}
                            onChange={handleChange}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none"
                            required
                        />
                    </div>

                    {/* Message Field */}
                    <div className="mb-4">
                        <label htmlFor="message" className="block text-sm font-medium text-gray-700">
                            What can we help you with?
                        </label>
                        <textarea
                            name="message"
                            value={formData.message}
                            onChange={handleChange}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2 focus:outline-none"
                            rows="3"
                        />
                    </div>

                    {/* Submit Button */}
                    <div>
                        <button
                            type="submit"
                            className="w-full bg-customGreen text-white  py-2 px-4 rounded-md hover:bg-customDarkGreen focus:outline-none"
                        >
                            Submit
                        </button>
                    </div>
                </form>
            </div>


        </div>
    )
}
